from torch.cuda.amp import autocast as autocast
from torchvision.utils import make_grid

from networks.losses import *
from networks.scene_text_recognition.cal_loss import * 
from tools.utils import requires_grad
from tools.eval import cal_metric


def clip_grad(model):
    for h in model.parameters():
        h.data.clamp_(-0.01, 0.01)

def cal_D_loss_srnet(G, D1, D2, data, config):
    i_t, i_s, t_sk, t_t, t_b, t_f, mask_t = data
    o_sk, o_t, o_b, o_f = G(i_t, i_s, (i_t.shape[2], i_t.shape[3]))
    #???
    #K = torch.nn.ZeroPad2d((0, 1, 1, 0))
    #o_sk = K(o_sk)
    #o_t = K(o_t)
    #o_b = K(o_b)
    #o_f = K(o_f)
    i_db_true = torch.cat((t_b, i_s), dim = 1)
    i_db_pred = torch.cat((o_b, i_s), dim = 1)
    i_df_true = torch.cat((t_f, i_t), dim = 1)
    i_df_pred = torch.cat((o_f, i_t), dim = 1)
    o_db_true = D1(i_db_true)
    o_db_pred = D1(i_db_pred)
    o_df_true = D2(i_df_true)
    o_df_pred = D2(i_df_pred)
    db_loss = build_discriminator_loss(o_db_true,  o_db_pred, config.loss_params)
    df_loss = build_discriminator_loss(o_df_true, o_df_pred, config.loss_params)
    return db_loss, df_loss

def cal_D_loss_imgur5k(G, D1, D2, data, config):
    i_t, i_s, t_s, source_text, target_text, = data
    o_s_t, o_t_t, o_b_t, o_f_t = G(i_t, i_s, (i_t.shape[2], i_t.shape[3]))
    o_s_s, o_t_s, o_b_s, o_f_s = G(t_s, i_s, (t_s.shape[2], t_s.shape[3]))
    i_df_true = torch.cat((i_s, t_s), dim = 1)
    i_df_pred = torch.cat((o_f_s, t_s), dim = 1)
    o_df_true = D1(i_df_true)
    o_df_pred = D1(i_df_pred)

    df_loss = build_discriminator_loss(o_df_true,  o_df_pred, config.loss_params)
    return torch.tensor(0.0), df_loss

def cal_G_loss_srnet(G, D1, D2, vgg_features, recognition, recognition_criterion, device, data, config):
    i_t, i_s, t_sk, t_t, t_b, t_f, mask_t = data
    o_sk, o_t, o_b, o_f = G(i_t, i_s, (i_t.shape[2], i_t.shape[3]))
    #???
    #K = torch.nn.ZeroPad2d((0, 1, 1, 0))
    #o_sk = K(o_sk)
    #o_t = K(o_t)
    #o_b = K(o_b)
    #o_f = K(o_f)
    i_db_true = torch.cat((t_b, i_s), dim = 1)
    i_db_pred = torch.cat((o_b, i_s), dim = 1)
    i_df_true = torch.cat((t_f, i_t), dim = 1)
    i_df_pred = torch.cat((o_f, i_t), dim = 1)
    o_db_pred = D1(i_db_pred)
    o_df_pred = D2(i_df_pred)
    
    i_vgg = torch.cat((t_f, o_f), dim = 0)
    out_vgg = vgg_features(i_vgg)

    out_g = [o_sk, o_t, o_b, o_f, mask_t]    
    out_d = [o_db_pred, o_df_pred]
    labels = [t_sk, t_t, t_b, t_f]

    g_loss, detail = build_generator_loss_srnet(out_g, out_d, out_vgg, labels, config.loss_params)
    return g_loss, detail, out_g

def cal_G_loss_imgur5k(G, D1, D2, vgg_features, recognition, recognition_criterion, device, data, config):
    i_t, i_s, t_s, source_text, target_text = data
    o_s_t, o_t_t, o_b_t, o_f_t = G(i_t, i_s, (i_t.shape[2], i_t.shape[3]))
    o_s_s, o_t_s, o_b_s, o_f_s = G(t_s, i_s, (t_s.shape[2], t_s.shape[3]))

    o_s_ss, o_t_ss, o_b_ss, o_f_ss = G(t_s, o_f_s, (t_s.shape[2], t_s.shape[3]))
    #o_s_ts, o_t_ts, o_b_ts, o_f_ts = G(t_s, o_f_t, (t_s.shape[2], t_s.shape[3]))
    #o_s_st, o_t_st, o_b_st, o_f_st = G(i_t, o_f_s, (i_t.shape[2], i_t.shape[3]))
    #o_s_tt, o_t_tt, o_b_tt, o_f_tt = G(i_t, o_f_t, (i_t.shape[2], i_t.shape[3]))

   
    i_df_pred = torch.cat((o_f_s, t_s), dim = 1)
    o_df_pred = D1(i_df_pred)

    i_vgg = torch.cat((i_s, o_f_s), dim = 0)
    out_vgg = vgg_features(i_vgg)

    fs_tc_loss = text_content_loss(o_f_s, source_text, device, recognition_criterion, criterion)
    ft_tc_loss = text_content_loss(o_f_t, target_text, device, recognition_criterion, criterion)
    ts_tc_loss = text_content_loss(o_t_s, source_text, device, recognition_criterion, criterion)
    tt_tc_loss = text_content_loss(o_t_t, target_text, device, recognition_criterion, criterion)
    l_f_tc = fs_tc_loss + ft_tc_loss + ts_tc_loss + tt_tc_loss

    out_g = [o_f_s, o_f_ss]    
    out_d = [o_df_pred]
    label = [i_s]

    out = [o_s_t, o_t_t, o_b_t, o_f_t] + [o_s_s, o_t_s, o_b_s, o_f_s] + [o_s_ss, o_t_ss, o_b_ss, o_f_ss]

    g_loss, detail = build_generator_loss_imgur5k(out_g, out_d, out_vgg, labels, config.loss_params)
    return g_loss+tc_loss, detail+[l_f_tc], out

def train_D_step_srnet(current_step, data, G, D1, D2, G_optimizer, D1_optimizer, D2_optimizer, device, config, scaler):
    requires_grad(G, False)
    requires_grad(D1, True)
    requires_grad(D2, True)

    i_t, i_s, t_sk, t_t, t_b, t_f, mask_t, img_name = data
    
    i_t = i_t.to(device)
    i_s = i_s.to(device)
    t_sk = t_sk.to(device)
    t_t = t_t.to(device)
    t_b = t_b.to(device)
    t_f = t_f.to(device)
    mask_t = mask_t.to(device)

    data = [i_t, i_s, t_sk, t_t, t_b, t_f, mask_t]

    if config.use_amp:# 前向过程(model + loss)开启 autocast
        with autocast(): 
            db_loss, df_loss = cal_D_loss_srnet(G, D1, D2, data, config)
    else:
        db_loss, df_loss = cal_D_loss_srnet(G, D1, D2, data, config)

    D1_optimizer.zero_grad()
    D2_optimizer.zero_grad()
    if config.use_amp and scalar is not None:
        scaler.scale(db_loss).backward()
        scaler.scale(df_loss).backward()
        scaler.step(D1_optimizer)
        scaler.step(D2_optimizer)
        scalar.update()
    else:
        db_loss.backward()
        df_loss.backward()
        #if config.parallel_type == 'DataParallel':
        #    D1_optimizer.module.step() 
        #    D2_optimizer.module.step()
        #else:
        D1_optimizer.step()
        D2_optimizer.step()
    clip_grad(D1)
    clip_grad(D2)
    return db_loss, df_loss

def train_D_step_imgur5k(current_step, data, G, D1, D2, G_optimizer, D1_optimizer, D2_optimizer, device, config, scaler):
    requires_grad(G, False)
    requires_grad(D1, True)
    requires_grad(D2, True)

    i_t, i_s, t_s, source_text, target_text, img_name = data
    
    i_t = i_t.to(device)
    i_s = i_s.to(device)
    t_s = t_s.to(device)

    data = [i_t, i_s, t_s, source_text, target_text]

    if config.use_amp:# 前向过程(model + loss)开启 autocast
        with autocast(): 
            _, df_loss = cal_D_loss_imgur5k(G, D1, D2, data, config)
    else:
        _, df_loss = cal_D_loss_imgur5k(G, D1, D2, data, config)

    D1_optimizer.zero_grad()
    if config.use_amp and scalar is not None:
        scaler.scale(df_loss).backward()
        scaler.step(D1_optimizer)
        scalar.update()
    else:
        df_loss.backward()
        D1_optimizer.step()
    clip_grad(D1)
    return torch.tensor(0.0), df_loss

def train_G_step_srnet(current_step, data, G, D1, D2, vgg_features, G_optimizer, D1_optimizer, D2_optimizer, recognition, recognition_criterion, \
device, config, scaler):
    requires_grad(G, True)
    requires_grad(D1, False)
    requires_grad(D2, False)
    
    i_t, i_s, t_sk, t_t, t_b, t_f, mask_t, img_name = data

    i_t = i_t.to(device)
    i_s = i_s.to(device)
    t_sk = t_sk.to(device)
    t_t = t_t.to(device)
    t_b = t_b.to(device)
    t_f = t_f.to(device)
    mask_t = mask_t.to(device)

    data = [i_t, i_s, t_sk, t_t, t_b, t_f, mask_t]
    
    if config.use_amp:# 前向过程(model + loss)开启 autocast
        with autocast(): 
            g_loss, other_loss, out_g = cal_G_loss_srnet(G, D1, D2, vgg_features, recognition, recognition_criterion, device, data, config)
    else:
        g_loss, other_loss, out_g = cal_G_loss_srnet(G, D1, D2, vgg_features, recognition, recognition_criterion, device, data, config)

    G_optimizer.zero_grad()
    if config.use_amp and scalar is not None:
        scaler.scale(g_loss).backward()
        scaler.step(G_optimizer)
        scaler.update()
    else:
        g_loss.backward()
        #if config.parallel_type == 'DataParallel':
        #    G_optimizer.module.step() 
        #else:
        G_optimizer.step()
    #clip_grad(G)
    return g_loss, other_loss

def train_G_step_imgur5k(current_step, data, G, D1, D2, vgg_features, G_optimizer, D1_optimizer, D2_optimizer, recognition, recognition_criterion, \
device, config, scaler):
    requires_grad(G, True)
    requires_grad(D1, False)
    requires_grad(D2, False)
    
    i_t, i_s, t_s, source_text, target_text, img_name = data

    i_t = i_t.to(device)
    i_s = i_s.to(device)
    t_s = t_s.to(device)
   
    data = [i_t, i_s, t_s, source_text, target_text]
    
    if config.use_amp:# 前向过程(model + loss)开启 autocast
        with autocast(): 
            g_loss, other_loss, out_g = cal_G_loss_imgur5k(G, D1, D2, vgg_features, recognition, recognition_criterion, device, data, config)
    else:
        g_loss, other_loss, out_g = cal_G_loss_imgur5k(G, D1, D2, vgg_features, recognition, recognition_criterion, device, data, config)

    G_optimizer.zero_grad()
    if config.use_amp and scalar is not None:
        scaler.scale(g_loss).backward()
        scaler.step(G_optimizer)
        scaler.update()
    else:
        g_loss.backward()
        #if config.parallel_type == 'DataParallel':
        #    G_optimizer.module.step() 
        #else:
        G_optimizer.step()
    #clip_grad(G)
    return g_loss, other_loss

def train_G_step(current_step, data, G, D1, D2, vgg_features, G_optimizer, D1_optimizer, D2_optimizer, recognition, recognition_criterion, \
device, config, scaler):
    if config.data_type == 'data_srnet':
        return train_G_step_srnet(current_step, data, G, D1, D2, vgg_features, G_optimizer, D1_optimizer, D2_optimizer, recognition, recognition_criterion, \
device, config, scaler)
    if config.data_type == 'data_imgur5k':
        return train_G_step_imgur5k(current_step, data, G, D1, D2, vgg_features, G_optimizer, D1_optimizer, D2_optimizer, recognition, recognition_criterion, \
device, config, scaler)

def train_D_step(current_step, data, G, D1, D2, G_optimizer, D1_optimizer, D2_optimizer, device, config, scaler):
    if config.data_type == 'data_srnet':
        return train_D_step_srnet(current_step, data, G, D1, D2, G_optimizer, D1_optimizer, D2_optimizer, device, config, scaler)
    if config.data_type == 'data_imgur5k':
        return train_D_step_imgur5k(current_step, data, G, D1, D2, G_optimizer, D1_optimizer, D2_optimizer, device, config, scaler)

    