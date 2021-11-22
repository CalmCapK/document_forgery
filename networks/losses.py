import torch

def build_dice_loss(x_t, x_o, params):   
    iflat = x_o.view(-1)
    tflat = x_t.view(-1)
    intersection = (iflat*tflat).sum()
    return 1. - torch.mean((2. * intersection + params['epsilon'])/(iflat.sum() +tflat.sum()+ params['epsilon']))

def build_l1_loss(x_t, x_o):
    return torch.mean(torch.abs(x_t - x_o))

#???
def build_l1_loss_with_mask(x_t, x_o, mask): 
    mask_ratio = 1. - mask.view(-1).sum() / torch.size(mask)
    l1 = torch.abs(x_t - x_o)
    return mask_ratio * torch.mean(l1 * mask) + (1. - mask_ratio) * torch.mean(l1 * (1. - mask))

def build_gan_loss(x_pred, params):
    gen_loss = -torch.mean(torch.log(torch.clamp(x_pred, params['epsilon'], 1.0)))
    return gen_loss

def build_perceptual_loss(x):        
    l = []
    for i, f in enumerate(x):
        l.append(build_l1_loss(f[0], f[1]))
    l = torch.stack(l, dim = 0)
    l = l.sum()
    return l

def build_gram_matrix(x):
    x_shape = x.shape
    c, h, w = x_shape[1], x_shape[2], x_shape[3]
    matrix = x.view((-1, c, h * w))
    matrix1 = torch.transpose(matrix, 1, 2)
    gram = torch.matmul(matrix, matrix1) / (h * w * c)
    return gram

def build_style_loss(x):
    l = []
    for i, f in enumerate(x):
        f_shape = f[0].shape[0] * f[0].shape[1] *f[0].shape[2]
        f_norm = 1. / f_shape
        gram_true = build_gram_matrix(f[0])
        gram_pred = build_gram_matrix(f[1])
        l.append(f_norm * (build_l1_loss(gram_true, gram_pred)))
    l = torch.stack(l, dim = 0)
    l = l.sum()
    return l

def build_vgg_loss(x):    
    splited = []
    for i, f in enumerate(x):
        splited.append(torch.chunk(f, 2))
    l_per = build_perceptual_loss(splited)
    l_style = build_style_loss(splited)
    return l_per, l_style

def build_discriminator_loss(x_true, x_fake, params):
    d_loss = -torch.mean(torch.log(torch.clamp(x_true, params['epsilon'], 1.0)) + torch.log(torch.clamp(1.0 - x_fake, params['epsilon'], 1.0)))
    return d_loss

def build_generator_loss_srnet(out_g, out_d, out_vgg, labels, params):
    o_sk, o_t, o_b, o_f, mask_t = out_g
    o_db_pred, o_df_pred = out_d
    o_vgg = out_vgg
    t_sk, t_t, t_b, t_f = labels
    #skeleton loss
    l_t_sk = params['lt_alpha'] * build_dice_loss(t_sk, o_sk, params) #0.7-0.8
    l_t_l1 = build_l1_loss(t_t, o_t)  #0.05-0.1
    l_t =  l_t_l1 + l_t_sk
    #Background Inpainting module loss
    l_b_gan = build_gan_loss(o_db_pred, params)  #4.61
    l_b_l1 = params['lb_beta'] * build_l1_loss(t_b, o_b)  #0.2-0.6
    l_b = l_b_gan + l_b_l1
    # fusion module loss
    l_f_gan = build_gan_loss(o_df_pred, params) #4.61
    l_f_l1 = params['lf_theta_1']* build_l1_loss(t_f, o_f) #0.7-1.5
    l_f_vgg_per, l_f_vgg_style = build_vgg_loss(o_vgg) 
    l_f_vgg_per = params['lf_theta_2'] * l_f_vgg_per  #1.8-2.5
    l_f_vgg_style = params['lf_theta_3'] * l_f_vgg_style  #1e5
    l_f = l_f_gan + l_f_vgg_per + l_f_vgg_style + l_f_l1
    
    l = params['lt'] * l_t + params['lb'] * l_b + params['lf'] * l_f
    return l, [l_t_sk, l_t_l1, l_b_gan, l_b_l1, l_f_gan, l_f_l1, l_f_vgg_per, l_f_vgg_style]


def build_generator_loss_imgur5k(out_g, out_d, out_vgg, labels, params):
    o_f_s, o_f_ss = out_g
    o_df_pred = out_d
    o_vgg = out_vgg
    i_s = labels

    l_f_gan = build_gan_loss(o_df_pred, params) #1.0  1.0
    l_f_l1 = params['lf_theta_1']*build_l1_loss(i_s, o_f_s)  #10 10 
    l_f_cyc = build_l1_loss(i_s, o_f_ss)  #none  1.0
    l_f_vgg_per, l_f_vgg_style = build_vgg_loss(o_vgg) 
    l_f_vgg_per = params['lf_theta_2'] * l_f_vgg_per  #1.0 1.0
    l_f_vgg_style = params['lf_theta_3'] * l_f_vgg_style  #500 500

    l = l_f_gan + l_f_vgg_per + l_f_vgg_style + l_f_l1 + l_f_cyc

    return l, [l_f_gan, l_f_l1, l_f_cyc, l_f_vgg_per, l_f_vgg_style]