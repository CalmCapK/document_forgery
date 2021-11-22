import torch

temp_shape = (0,0)

def calc_padding(h, w, k, s):
    h_pad = (((h-1)*s) + k - h)//2 
    w_pad = (((w-1)*s) + k - w)//2
    return (h_pad, w_pad)

def calc_inv_padding(h, w, k, s):
    h_pad = (k-h + ((h-1)*s))//2
    w_pad = (k-w + ((w-1)*s))//2
    return (h_pad, w_pad)

class Conv_bn_block(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._conv = torch.nn.Conv2d(*args, **kwargs)
        self._bn = torch.nn.BatchNorm2d(kwargs['out_channels'])  
    def forward(self, input):
        return torch.nn.functional.leaky_relu(self._bn(self._conv(input)),negative_slope=0.2)

class Res_block(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self._conv1 = torch.nn.Conv2d(in_channels, in_channels//4, kernel_size = 1, stride =1)
        self._conv2 = torch.nn.Conv2d(in_channels//4, in_channels//4, kernel_size = 3, stride = 1, padding = 1)
        self._conv3 = torch.nn.Conv2d(in_channels//4, in_channels, kernel_size = 1, stride=1)     
        self._bn = torch.nn.BatchNorm2d(in_channels)
       
    def forward(self, x):
        xin = x
        x = torch.nn.functional.leaky_relu(self._conv1(x),negative_slope=0.2)
        x = torch.nn.functional.leaky_relu(self._conv2(x),negative_slope=0.2)
        x = self._conv3(x)
        x = torch.add(xin, x)
        x = torch.nn.functional.leaky_relu(self._bn(x),negative_slope=0.2)
        return x

# conv+BN+leaky_relu *2
#((pool+leaky_relu)+(con+BN+leaky_relu)*2)*3
class encoder_net(torch.nn.Module):   
    def __init__(self, in_channels, get_feature_map = False):
        super().__init__()
        self.cnum = 32
        self.get_feature_map = get_feature_map
        self._conv1_1 = Conv_bn_block(in_channels = in_channels, out_channels = self.cnum, kernel_size = 3, stride = 1, padding = 1)  
        self._conv1_2 = Conv_bn_block(in_channels =self.cnum, out_channels = self.cnum, kernel_size = 3, stride = 1, padding = 1)
        #---------------------------
        self._pool1 = torch.nn.Conv2d(in_channels = self.cnum, out_channels = 2 * self.cnum, kernel_size = 3, stride = 2, padding = calc_padding(temp_shape[0], temp_shape[1], 3, 2))   
        #self._pool1 = torch.nn.Conv2d(in_channels = self.cnum, out_channels = 2 * self.cnum, kernel_size = 3, stride = 2, padding = 1)  
        self._conv2_1 = Conv_bn_block(in_channels = 2* self.cnum, out_channels = 2*self.cnum, kernel_size = 3, stride = 1, padding = 1)        
        self._conv2_2 = Conv_bn_block(in_channels = 2* self.cnum, out_channels = 2*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        #---------------------------
        self._pool2 = torch.nn.Conv2d(in_channels = 2*self.cnum, out_channels = 4* self.cnum, kernel_size = 3, stride = 2, padding = calc_padding(temp_shape[0], temp_shape[1], 3, 2))
        #self._pool2 = torch.nn.Conv2d(in_channels = 2*self.cnum, out_channels = 4* self.cnum, kernel_size = 3, stride = 2, padding = 1)
        self._conv3_1 = Conv_bn_block(in_channels = 4* self.cnum, out_channels = 4*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        self._conv3_2 = Conv_bn_block(in_channels = 4* self.cnum, out_channels = 4*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        #---------------------------
        self._pool3 = torch.nn.Conv2d(in_channels = 4*self.cnum, out_channels = 8* self.cnum, kernel_size = 3, stride = 2, padding = calc_padding(temp_shape[0], temp_shape[1], 3, 2)) 
        #self._pool3 = torch.nn.Conv2d(in_channels = 4*self.cnum, out_channels = 8* self.cnum, kernel_size = 3, stride = 2, padding = 1) 
        self._conv4_1 = Conv_bn_block(in_channels = 8* self.cnum, out_channels = 8*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        self._conv4_2 = Conv_bn_block(in_channels = 8* self.cnum, out_channels = 8*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
    def forward(self, x):
        x = self._conv1_1(x)
        x = self._conv1_2(x)
        x = torch.nn.functional.leaky_relu(self._pool1(x),negative_slope=0.2)
        #print(x.shape)
        x = self._conv2_1(x)
        x = self._conv2_2(x)
        f1 = x
        x = torch.nn.functional.leaky_relu(self._pool2(x),negative_slope=0.2)
        #print(x.shape)
        x = self._conv3_1(x)
        x = self._conv3_2(x)
        f2 = x
        x = torch.nn.functional.leaky_relu(self._pool3(x),negative_slope=0.2)
        x = self._conv4_1(x)
        x = self._conv4_2(x)
        
        if self.get_feature_map:
            return x, [f2, f1]
        else:
            return x

class build_res_block(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self._block1 = Res_block(in_channels)
        self._block2 = Res_block(in_channels)
        self._block3 = Res_block(in_channels)
        self._block4 = Res_block(in_channels)
        
    def forward(self, x): 
        x = self._block1(x)
        x = self._block2(x)
        x = self._block3(x)
        x = self._block4(x)
        return x

#con+BN+leaky_relu *2
#((ConvTranspose2d+leaky_relu)+(con+BN+leaky_relu)*2)*3
class decoder_net(torch.nn.Module):
    def __init__(self, in_channels, get_feature_map = False, mt=1, fn_mt=1):
        super().__init__()
        self.cnum = 32
        self.get_feature_map = get_feature_map
        
        self._conv1_1 = Conv_bn_block(in_channels = fn_mt*in_channels , out_channels = 8*self.cnum, kernel_size = 3, stride =1, padding = 1) 
        self._conv1_2 = Conv_bn_block(in_channels = 8*self.cnum, out_channels = 8*self.cnum, kernel_size = 3, stride =1, padding = 1)
        #-----------------
        self._deconv1 = torch.nn.ConvTranspose2d(8*self.cnum, 4*self.cnum, kernel_size = 3, stride = 2, padding = calc_inv_padding(temp_shape[0], temp_shape[1], 3, 2)) 
        self._conv2_1 = Conv_bn_block(in_channels = int(fn_mt*mt)*4*self.cnum, out_channels = 4*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        self._conv2_2 = Conv_bn_block(in_channels = 4*self.cnum, out_channels = 4*self.cnum, kernel_size = 3, stride = 1, padding = 1)     
        #-----------------
        self._deconv2 = torch.nn.ConvTranspose2d(4*self.cnum, 2*self.cnum, kernel_size =3 , stride = 2, padding = calc_inv_padding(temp_shape[0], temp_shape[1], 3, 2))
        self._conv3_1 = Conv_bn_block(in_channels = int(fn_mt*mt)*2*self.cnum, out_channels = 2*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        self._conv3_2 = Conv_bn_block(in_channels = 2*self.cnum, out_channels = 2*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        #----------------
        self._deconv3 = torch.nn.ConvTranspose2d(2*self.cnum, self.cnum, kernel_size =3 , stride = 2, padding = calc_inv_padding(temp_shape[0], temp_shape[1], 3, 2))
        self._conv4_1 = Conv_bn_block(in_channels = self.cnum, out_channels = self.cnum, kernel_size = 3, stride = 1, padding = 1)
        self._conv4_2 = Conv_bn_block(in_channels = self.cnum, out_channels = self.cnum, kernel_size = 3, stride = 1, padding = 1)

        
    def forward(self, x, fuse = None):
        '''
        fuse
        torch.Size([32, 256, 7, 30])
        torch.Size([32, 128, 15, 61])
        torch.Size([32, 64, 31, 123])
        '''
        if fuse:
            if fuse[0] == None:
                print(fuse[1].shape, fuse[2].shape)
            else:
                print(fuse[0].shape, fuse[1].shape, fuse[2].shape)
        if fuse and fuse[0] is not None:
            x = torch.cat((x, fuse[0]), dim = 1)
        x = self._conv1_1(x)
        x = self._conv1_2(x)
        f1 = x
        #print(x.shape)
        x = torch.nn.functional.leaky_relu(self._deconv1(x), negative_slope=0.2)
        #print(x.shape)
        if fuse and fuse[1] is not None:
            x = torch.cat((x, fuse[1]), dim = 1) 
        x = self._conv2_1(x)
        x = self._conv2_2(x)
        f2 = x
        #print(x.shape)
        x = torch.nn.functional.leaky_relu(self._deconv2(x), negative_slope=0.2)
        #print(x.shape)
        if fuse and fuse[2] is not None:
            x = torch.cat((x, fuse[2]), dim = 1)
        x = self._conv3_1(x)
        x = self._conv3_2(x)
        f3 = x
        #print(x.shape)
        x = torch.nn.functional.leaky_relu(self._deconv3(x), negative_slope=0.2)
        #print(x.shape)
        x = self._conv4_1(x)
        x = self._conv4_2(x)

        if self.get_feature_map:
            return x, [f1, f2, f3]        
        else:
            return x


class Generator(torch.nn.Module):
    def __init__(self, in_channels):       
        super().__init__()
        self.cnum = 32
        self._encoder = encoder_net(2*in_channels, get_feature_map = True)
        self._res = build_res_block(8*self.cnum)
        self._conv_tsb = Conv_bn_block(in_channels = 8* self.cnum, out_channels = 8*3*self.cnum, kernel_size = 3, stride = 1, padding = 1)
    
        self._decoder_b = decoder_net(8*self.cnum,  get_feature_map = True, mt=2)
        self._out_b = torch.nn.Conv2d(self.cnum, 3, kernel_size = 3, stride = 1, padding = 1)

        #self._decoder_s = decoder_net(8*self.cnum, get_feature_map = True, fn_mt = 2, mt=1.5)
        self._decoder_s = decoder_net(8*self.cnum, get_feature_map = True)
        self._out_s = torch.nn.Conv2d(self.cnum, 3, kernel_size = 3, stride = 1, padding = 1)
         
        self._decoder_t = decoder_net(8*self.cnum, get_feature_map = True)
        #self._decoder_t = decoder_net(8*self.cnum, get_feature_map = True, fn_mt = 2, mt=1.5)
        #self._out_mask = torch.nn.Conv2d(self.cnum, 1, kernel_size =3, stride = 1, padding = 1)
        self._out_t = torch.nn.Conv2d(self.cnum, 3, kernel_size = 3, stride = 1, padding = 1)
    
        self._decoder = decoder_net(8*self.cnum, get_feature_map = False, fn_mt = 3)
        self._out = torch.nn.Conv2d(self.cnum, 3, kernel_size = 3, stride = 1, padding = 1)
    

    def forward(self, i_t, i_s, gbl_shape):
        #shape: [32, 3, 63, 247]
        temp_shape = gbl_shape #???
        x = torch.cat((i_t, i_s), dim = 1)
        x, f_encoder = self._encoder(x) #[32, 256, 7, 30]
        x = self._res(x) #[32, 256, 7, 30]
        x = self._conv_tsb(x) #[32, 256*3, 7, 30]
        x_t, x_s, x_b = torch.split(x, 8*self.cnum, dim=1)

        y_b, f_b = self._decoder_b(x_b, fuse = [None] + f_encoder) #[32, 32, 63, 247]
        y_b = torch.tanh(self._out_b(y_b)) #[32, 3, 63, 247]
        #f_encoder_b = []
        #for i in range(len(f_b)):
        #    if i == 0:
        #        f_encoder_b.append(f_b[i])
        #    else:
        #        f_encoder_b.append(torch.cat((f_encoder[i], f_b[i]), dim = 1))
        
        #y_s, f_s = self._decoder_s(x_s, fuse = f_encoder_b) #[32, 32, 63, 247]
        y_s, f_s = self._decoder_s(x_s, fuse = None) #[32, 32, 63, 247]
        y_s = torch.tanh(self._out_s(y_s)) #[32, 3, 63, 247]
        #f_encoder_s = []
        #for i in range(len(f_s)):
        #    if i == 0:
        #        f_encoder_s.append(f_s[i])
        #    else:
        #        f_encoder_s.append(torch.cat((f_encoder[i], f_s[i]), dim = 1))

        #y_t_mask, f_t_mask = self._decoder_t(x_t, fuse = f_encoder_s)  #[32, 32, 63, 247]
        y_t_mask, f_t_mask = self._decoder_t(x_t, fuse = None)
        #y_mask = torch.sigmoid(self._out_mask(y_t_mask)) #[32, 1, 63, 247]
        y_t = torch.tanh(self._out_t(y_t_mask)) #[32, 3, 63, 247]

        f_encoder_s = []
        for i in range(len(f_s)):
            if i == 0:
                f_encoder_s.append(f_s[i])
            else:
                f_encoder_s.append(torch.cat((f_encoder[i-1], f_s[i]), dim = 1))
        x_t_b = torch.cat((x_t, x_b), dim = 1)  #[32, 256*2, 7, 30]
        y_f = self._decoder(x_t_b, fuse=f_encoder_s) #[32, 32, 63, 247]
        y_f = torch.tanh(self._out(y)) #[32, 3, 63, 247]
        return y_s, y_t, y_b, y_f
