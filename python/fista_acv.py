# fista-acv
# Patricio Guerrero
# KU Leuven
# patricio.guerrero@kuleuven.be


import torch
import numpy
import tomosipo as sipo



def vectors_tomosipo(pixel, voxel, sod, sdd, rot_step, angles, det_x, det_y, eta):

    # define geometry for tomosipo
    # eta theta phi in degrees
    
    pixel /= voxel 
    angles_seq = (numpy.arange(angles) * rot_step + 0.) 
    angles_seq *= numpy.pi / 180  
    sod /= voxel
    sdd /= voxel
    odd = sdd - sod  
    mgn_i = sod / sdd
    
    det_x /= voxel
    det_y /= voxel 

    eta *= numpy.pi / 180
    theta = 0; phi = 0    # 
    theta *= numpy.pi / 180 
    phi *= numpy.pi / 180

    sangles = numpy.sin(angles_seq)
    cangles = numpy.cos(angles_seq)

    # rotation matrices
    
    def ss(a):
        return numpy.sin(a)
    def cc(a):
        return numpy.cos(a)
    def rot_eta(a):
        return numpy.array([[1,0,0],[0,cc(a),-ss(a)],[0,ss(a),cc(a)]])
    def rot_theta(a):
        return numpy.array([[cc(a),0,-ss(a)],[0,1,0],[ss(a),0,cc(a)]])
    def rot_phi(a):
        return numpy.array([[cc(a),-ss(a),0],[ss(a),cc(a),0],[0,0,1]])

    u_shift = sangles * det_x
    v_shift = cangles * det_x

    rot_det = rot_theta(theta) @ rot_phi(phi) @ rot_eta(eta)

    det_u = rot_det @ numpy.array([0, pixel, 0])
    det_v = rot_det @ numpy.array([0, 0, pixel]) 

    vectors = numpy.zeros((angles, 12))
    # (source, detector center, det01, det10)
    
    vectors[:,0] = cangles * sod     
    vectors[:,1] = -sangles * sod     
    vectors[:,2] = 0
    vectors[:,3] = -cangles * odd + u_shift    
    vectors[:,4] = sangles * odd + v_shift
    vectors[:,5] = det_y
    vectors[:,6] = cangles * det_u[0] + sangles * det_u[1]
    vectors[:,7] = -sangles * det_u[0] + cangles * det_u[1]
    vectors[:,8] = det_u[2]
    vectors[:,9] = cangles * det_v[0] + sangles * det_v[1]
    vectors[:,10] = -sangles * det_v[0] + cangles * det_v[1]
    vectors[:,11] = det_v[2]

    return vectors




def positive(u):

    #projection to positive orthant (component-wise)
    return torch.clamp(u, min = 0)


def ct_operators(pixels, FOV, slices, slices_data, vectors, y_pix, magn):

    # tomographic forward and backprojection
    # for a given tomosipo geometry in vectors
    # y_pix is the middle vertical pixel coordinate from [-pixels/2, pixels/2]
    # let 0 if you need to reconstruct around the middle region (vertically)
    

    vol_geom = sipo.volume_vec(shape = (FOV, FOV, slices), pos = (0/pixels,0/pixels,y_pix/pixels), w=(1/pixels, 0, 0), v=(0, 1/pixels, 0), u=(0, 0, 1/pixels))   #400, 150
    cone_geom = sipo.cone_vec(shape = (slices_data, pixels), src_pos= vectors[:,0:3]/pixels, det_pos=(vectors[:,3:6]+numpy.array([0,0,y_pix*magn]))/pixels, det_v = vectors[:,9:12]/pixels, det_u = vectors[:,6:9]/pixels)
    
    A = sipo.operator(vol_geom, cone_geom)

    dev = torch.device('cuda:0')
    xx = numpy.linspace(-1, 1, FOV)
    xx, yy = numpy.meshgrid(xx, xx)
    mask = xx*xx + yy*yy < 0.95
    mask = torch.from_numpy(mask).to(dev)

    fp = lambda X: A(X) 
    bp = lambda Y: torch.mul(A.T(Y), mask.reshape(FOV, FOV, 1))

    return fp, bp


def power_method(fp, bp, pixels, slices, dev):

    # to compute the norm of the operator bp (fp)
    # Sidky Jørgensen Pan 2012

    x = torch.ones((pixels, pixels, slices), device = dev) 
    for i in numpy.arange(20):
        x = bp(fp(x)) 
        x /= torch.sqrt((x*x).sum())

    cone = fp(x)
    return ((cone*cone).sum()).cpu().numpy()




def gradient3D(u):

    # forward differences gradient
    
    gx = torch.roll(u, -1, 0) - u
    gy = torch.roll(u, -1, 1) - u
    gz = torch.roll(u, -1, 2) - u
    gx[-1] = 0
    gy[:,-1] = 0
    gz[:,:,-1] = 0    
    
    return torch.stack((gx, gy, gz))



def divergence3D(u, v, z):

    # backward differences divergent

    gu = u - torch.roll(u, 1, 0) 
    gv = v - torch.roll(v, 1, 1)
    gz = z - torch.roll(z, 1, 2)
    gu[0] = u[0]+0
    gu[-1] = -u[-2]
    gv[:,0] = v[:,0]+0
    gv[:,-1] = -v[:,-2]
    gz[:,:,0] = z[:,:,0]+0
    gz[:,:,-1] = -z[:,:,-2]
    
    return gu + gv + gz


def prepare_data(data, slices, pixels, FOV, y_pix, sod, voxel):

    # pick the gpu
    dev = torch.device('cuda:0')

    # find rows in data to crop according to natterer-wubbeling eq (5.78)
    dir_top = sod/voxel/pixels / ( sod/voxel/pixels - FOV/2/pixels )
    dir_bot = sod/voxel/pixels / ( sod/voxel/pixels + FOV/2/pixels )
    if y_pix >= 0: y_pix1 = y_pix + slices//2
    else: y_pix1 = y_pix - slices//2
    v_top =  dir_top * (y_pix1) 
    v_bot =  dir_bot * (y_pix1)
    
    slices_data = int(abs(v_top - y_pix)) + 1
    print('data crooped by', pixels//2 - slices_data + y_pix , pixels//2 + slices_data + y_pix)
    
    pixbot = max(0, pixels//2 - slices_data + y_pix) 
    pixtop = min(pixels, pixels//2 + slices_data + y_pix) 
    data = data[pixbot : pixtop]
   

    data = torch.from_numpy(data).to(dev)
    print('dataGPU', data.dtype, data.shape)
    
    
    return dev, data 




def fista_cv3D(data, FOV, vectors, slices, lambd, q, iters, y_pix, magn, dev):

    # use it to reconstruct after having the parameter estimated

    slices_data, views, pixels = data.shape

    x = torch.zeros((FOV, FOV, slices), device = dev)
    y = torch.zeros((FOV, FOV, slices), device = dev)

    fp, bp = ct_operators(pixels, FOV, slices, slices_data, vectors, y_pix, magn)
    
    tau = 1 / q
    t = 1

    xx = numpy.linspace(-1, 1, FOV)
    xx, yy = numpy.meshgrid(xx, xx)
    mask = xx*xx + yy*yy < 0.95
    mask = torch.from_numpy(mask).to(dev)
    
    torch.cuda.empty_cache()

    for i in  numpy.arange(iters):

        x1 = x + 0
        x = denoise_cv3D(y - tau * bp(fp(y) - data), lambd * tau, slices, FOV, dev)

        t1 = t - 1
        t = 0.5 * (1 + numpy.sqrt(1 + 4*t*t))
        y = x + t1/t * (x-x1)
        
        torch.cuda.empty_cache()

    return torch.mul(x, mask.reshape(FOV, FOV, 1)).cpu().numpy()



def fista_acv3D(phantom, data, FOV, vectors, slices, lambd, q, iters, y_pix, magn, dev):

    # to be used in the NGD algorthm to estimate lambda
    
    slices_data, views, pixels = data.shape

    x = torch.zeros((FOV, FOV, slices), device = dev)
    y = torch.zeros((FOV, FOV, slices), device = dev)

    fp, bp = ct_operators(pixels, FOV, slices, slices_data, vectors, y_pix, magn)
    
    tau = 1 / q
    t = 1
    
    xx = numpy.linspace(-1, 1, FOV)
    xx, yy = numpy.meshgrid(xx, xx)
    mask = xx*xx + yy*yy < 0.95
    mask = torch.from_numpy(mask).to(dev)

    torch.cuda.empty_cache()

    for i in  numpy.arange(iters):

        x1 = x + 0
        x = denoise_acv3D(y - tau * bp(fp(y) - data), lambd * tau, slices, FOV, dev)

        t1 = t - 1
        t = 0.5 * (1 + numpy.sqrt(1 + 4*t*t))
        y = x + t1/t * (x-x1)
        
        torch.cuda.empty_cache()

    return torch.mul(x, mask.reshape(FOV, FOV, 1))     




def denoise_acv3D(b, lambd, slices, FOV, dev):

    # aCV algorithm with tomosipo / pytorch for denoising
    # ud, vd: initial guess

    ud = torch.zeros((FOV, FOV, slices), device = dev)
    vd = torch.zeros((3, FOV, FOV, slices), device = dev)

    tau = 1
    sigma = 1 / (tau * 18)
    iters = 5

    torch.cuda.empty_cache()
    
    for i in  numpy.arange(iters):


        ud -= tau * ( -divergence3D(vd[0],vd[1],vd[2]) + ud - b)
        ud[ud<0] = 0

        vd += sigma * gradient3D(ud)
        vd = proj_2(vd, lambd, b)


    ud -= tau * ( -divergence3D(vd[0],vd[1],vd[2]) + ud - b)
    ud[ud<0] = 0
    
    return ud 


def denoise_cv3D(b, lambd, slices, FOV, dev):

    # CV algorithm with tomosipo / pytorch for denoising
    # ud, vd: initial guess

    
    ud = torch.zeros((FOV, FOV, slices), device = dev)
    vd = torch.zeros((3, FOV, FOV, slices), device = dev)
  
    tau = 1
    sigma = 1 / (tau * 18)
    lambd_i = 1 / lambd
    iters = 5

    for i in  numpy.arange(iters):
        
        ud -= tau * ( -divergence3D(vd[0],vd[1],vd[2]) + ud - b)
        ud[ud<0] = 0
        
        vd += sigma * gradient3D(ud)
        vd /= torch.clamp(lambd_i * torch.sqrt(vd[0]*vd[0] + vd[1]*vd[1] + vd[2]*vd[2]), min = 1)


    return positive(ud - tau * ( -divergence3D(vd[0],vd[1],vd[2]) + ud - b))


        


def hyperparameter_NGD(data, ground, vectors, slices, q, N, sod, sdd, voxel):

    # regularization parameter estimation based on a HQ ground truth scan
    # with automatic differentiation 
    
    # ground : ground truth HQ scan
    # vectors: tomosipo geometry 
    # lambd : initial value, > 0
    

    _, views, pixels = data.shape
    FOV = 1400
    print('FOV AD', FOV)
    dev, data  = prepare_data(data, slices, pixels, pixels, 0, sod, voxel)
    
    ground = numpy.transpose(ground, axes = (0,2,1))  #for tomosipo
    ground = numpy.moveaxis(ground, 0, -1)
        
    lambd = torch.tensor(1e-5, device = dev)  # initial guess    
    lam_y = lambd + 0   # nesterov variable
    n_GD = 40   # gradient descent iterations
    n_LS = 20   # line search iterations
    c = 0.01    # backtracking constant in page 33 nocedal-wright

    gamma = 1e-2 

    bound = 1
    t = 1  # nesterov

    magn = sdd / sod
    ground = ground[(pixels-FOV)//2:(pixels+FOV)//2, (pixels-FOV)//2:(pixels+FOV)//2]
    xx = numpy.linspace(-1, 1, FOV)
    xx, yy = numpy.meshgrid(xx, xx)
    mask = xx*xx + yy*yy < 0.95
    ground *= mask.reshape(FOV, FOV, 1)
    gnorm_i = 1/(ground * ground).sum()
    ground = torch.from_numpy(ground).to(dev)


    torch.cuda.empty_cache()    
    for k in numpy.arange(n_GD):  # gradient descent

        lambd.requires_grad_(True)
        lambd.grad = None

        ux = fista_acv3D(ground, data, FOV, vectors, slices, lambd, q, N, 0, magn, dev)
        lossGD = lossl2(ux, ground) * gnorm_i
        
        lossGD.backward(inputs = [lambd])
        grad = lambd.grad 
        lambd.requires_grad_(False)
        
        lossLS = lossGD + 0        

        # box constraints
        if grad * gamma >= (lambd-1e-6) : gamma = (lambd-1e-6) / grad  # >1e-6 constraint
        if -grad * gamma >= (bound-lambd) : gamma = -(bound-lambd) / grad # bound constraint

        print('******* GD', k, lambd, grad.cpu().numpy(), lossGD.detach().cpu().numpy())
        
        for l in numpy.arange(n_LS):  # backtracking LS
            
            gg = grad * gamma   
            if abs(gg) < 1e-8 :     
                print('abs(gg) < tol')
                return lambd.cpu().numpy()

            lambdLS = lambd - gg  
                        
            u = fista_acv3D(ground, data, FOV, vectors, slices, lambdLS, q, N, 0, magn, dev)
            u -= ground 
            lossLS = (u * u).sum() * 0.5 * gnorm_i

            if lossLS <= lossGD - c*gg*grad :   # Armijo condition
                break

            if l+1 == n_LS :
                print('max armijo iterations')
                return lambd.cpu().numpy() 

            if abs(lambdLS - lambd) < 1e-5:
                break  # stop if LS change is < tol 
            
            gamma *= 0.5   # contration factor 


        lambdGD = lambd + 0    

        # nesterov
        lambd = lam_y - gg
        t1 = t - 1
        t = 0.5 * (1 + numpy.sqrt(1 + 4*t*t))
        lam_y = lambd + t1/t * (lambd -  lambdGD)
        
        if abs(lambdGD - lambd) < 1e-6:
            print('diff GD < tol')
            break # stop if GD change is < tol
                
    return lambd.cpu().numpy()




class Lossl2(torch.autograd.Function):

    #squared l2 loss between ground-truth and some input 

    @staticmethod
    def forward(ctx, input, ground):
       
        diff = input - ground
        ctx.save_for_backward(diff)
        return (diff * diff).sum() * 0.5 

    @staticmethod
    def backward(ctx, grad_output):
        
        diff, = ctx.saved_tensors
        return grad_output * diff, None 

def lossl2(input, ground):
    return Lossl2.apply(input, ground)




class Proj_2(torch.autograd.Function):

    # projection onto 2-balls with radius lmb
    # with its derivative wrt lambda
    # (4.23) in chambolle-pock 2016   

    
    @staticmethod
    def forward(ctx, v, lmb, b):
        
        den =  torch.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) / lmb
        den[den < 1] = 1   
        ctx.save_for_backward(v.half(), lmb)
        return v / den

    
    @staticmethod
    def backward(ctx, grad_output):

        v, lmb = ctx.saved_tensors
        v = v.float()
        nor =  torch.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
        den = nor / lmb
        den[den < 1] = 1
        heav = (nor / lmb) > 1
        dlmb = v * nor * heav / (lmb * lmb * den * den)
   
        return None, grad_output * dlmb, None


def proj_2(v, lmb, b):
    return Proj_2.apply(v, lmb, b)


