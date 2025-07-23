import os
import torch
import numpy as np
import pandas
from torch import nn
import torch.nn.functional as F
import datetime
import timeit 
#from sklearn.preprocessing import scale

opt_str="adam" # adam simple
reltol = 1e-8 # if earlytop, this is used, has a huge impact on the results
stop_if_loss_incr_thrsh=50

loss_fn = nn.MSELoss(reduction='mean') 
learning_rate = 1e-4 # 1e-4, 1e-5 is too slow


def main(dat, opt_numCode, opt_seed, opt_model, opt_gpu, opt_k, opt_nEpochs, opt_constr, opt_tuneParam, opt_penfun, opt_ortho, opt_earlystop, verbose):        
    
    if torch.cuda.is_available() and opt_gpu >= 0:
        device = torch.device("cuda:"+str(opt_gpu))
    else :
        device = torch.device("cpu")        
    dat = torch.tensor(dat.values, dtype=torch.float, device=device) # dtype is needed here otherwise will get an error in training: Expected object of scalar type Float but got scalar type Double for argument #4 'mat1'
    
    N=dat.shape[0]; n=N
    p=dat.shape[1]        
    
    torch.manual_seed(opt_seed)
    out=[]
    x_new = dat
    allcodes=torch.zeros([n,0], dtype=torch.float, device=device)
    
    if opt_numCode<0: opt_numCode=p
    for i in range(opt_numCode) :
        if opt_model == 'o':
            if i>0 : x_new = x_new - out[i-1][2]
            
        out.append (AE1(x_new, i, allcodes, device, opt_model, opt_k, opt_nEpochs, opt_constr, opt_tuneParam, opt_penfun, opt_ortho, opt_earlystop, verbose))
        allcodes=torch.cat([allcodes, out[i][3]],1)
        
    return out



# find one nonlinear component
def AE1(x, stage, prev_codes, device, opt_model, opt_k, opt_nEpochs, opt_constr, opt_tuneParam, opt_penfun, opt_ortho, opt_earlystop, verbose):    
    
    p=x.shape[1]
    
    if opt_model == 'o':
        model = Model_Old(opt_k, p)
    elif opt_model == 'n':
        model = Model_New(opt_k, p, stage)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    y=x # y is output
    
    loss_old = np.inf
    cnt=0
    history=[]
    if verbose : print("iter", " reconstruct_loss", " penalized loss")
    
    for t in range(opt_nEpochs):
        if opt_model == 'o':
            y_pred, code = model(x)
        elif opt_model == 'n':
            y_pred, code = model(x, prev_codes)
            
        loss = loss_fn(y_pred, y)
        reconstruct_loss = loss.item()
        
        if opt_constr=="penalization":
            # penalize negtive gradients
            for j in range(p) :
                d = torch.autograd.grad(y_pred[:, j].sum(), code, retain_graph=True, create_graph=True)[0]
                loss += opt_tuneParam * F.relu(-d).sum() 
        
        elif opt_constr=="newpenalization":
            # in the first stage, manually compute derivatives at a grid of values in the range of the code
            # in the later stages, same as penalization
            d1 = model.output_code_deriv(stage==0, device)
            if opt_penfun == 'sum':
                loss += opt_tuneParam * F.relu(-d1).sum()
            elif opt_penfun == 'mean':
                loss += opt_tuneParam * F.relu(-d1).mean()
                
        elif opt_constr=="deltapenalization":
            # sort by code and take differences in y_pred
            order = torch.argsort(code[:,0])
            d = y_pred[order[1:]] - y_pred[order[:-1]]
            if opt_penfun == 'sum':
                loss += opt_tuneParam * F.relu(-d).sum()
            elif opt_penfun == 'mean':
                loss += opt_tuneParam * F.relu(-d).mean()
                                
        elif opt_constr=="hessian" :
            # penalize second order derivatives
            for j in range(p) :
                d = torch.autograd.grad(y_pred[:, j].sum(), code, retain_graph=True, create_graph=True)[0]
                h = torch.autograd.grad(d.sum(), code, retain_graph=True, create_graph=True)[0]
                # really difficult to implement integration because of sort
                # torch.pow(h,2).sort()
                loss += opt_tuneParam * torch.pow(h,2).sum() 
        
        if opt_ortho>0:
            # penalize covariance
            prev_means = [prev_codes[:,j].mean() for j in range(stage)]
            mean_code = code[:,0].mean()
            for j in range(stage) :
                loss += opt_ortho * abs((code[:,0] * prev_codes[:,j]).mean() - mean_code * prev_means[j])
        
      
        if t % 100 == 0:
            if verbose : print(t, reconstruct_loss, loss.item())
            history.append(loss.item())
        
        if opt_earlystop=="yes":
            if (abs(loss_old-loss.item())/loss_old<reltol):
                if verbose : print("reltol reached")
                break    
    
        if (loss.item()>loss_old) :
            cnt+=1
            if(cnt == stop_if_loss_incr_thrsh) :
                if verbose : print("loss starts increased for "+str(stop_if_loss_incr_thrsh) + " times")
                break
        else :
            cnt=0
        loss_old=loss.item()
            
        if (opt_str=="simple") :
            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad
        elif (opt_str=="adam") :
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (opt_constr=="constrained"):
                with torch.no_grad():
                    iter=0
                    for value in model.parameters():
                        iter += 1
                        if (iter==5 or iter==7):
                            value.data.clamp_(0)
    
    
#    from matplotlib import pyplot as plt
#    plt.figure(figsize=(30,8))
#    for i in range(4):
#        plt.subplot(2,4,i+1)
#        plt.scatter(code.detach().cpu().numpy(), y_pred[:,i-1].detach().cpu().numpy(), s=1)
#        plt.subplot(2,4,i+5)
#        plt.plot(range(d1.shape[1]), d1.detach().cpu().numpy()[i,])
#        plt.axhline(y=0, lw=0.5)
#    plt.savefig("btmp.pdf"); plt.close()
    
    
    decoder_w=torch.cat((model.demap[0].weight, model.output[0].weight.t()),1)
    decoder_b=torch.cat((model.demap[0].bias,   model.output[0].bias),0)
    # the following line matches y_pred
    #torch.mm((torch.mm(code, model.top[0].weight.t()) + model.top[0].bias).tanh(), model.top[2].weight.t()) + model.top[2].bias
    
    return reconstruct_loss, loss.item(), y_pred.detach(), code.detach(), decoder_w.detach().cpu(), decoder_b.detach().cpu(), history


class Model_New (nn.Module):
    def __init__(self, k, p, stage):
        super().__init__()
        self.bottom = nn.Sequential(nn.Linear(p, k), nn.Tanh(), nn.Linear(k, 1))
        #self.top = nn.Sequential(nn.Linear(stage+1, k), nn.Tanh(), nn.Linear(k, p))
        # top is split into demap and output
        self.demap  = nn.Sequential(nn.Linear(stage+1, k), nn.Tanh())
        self.output = nn.Sequential(nn.Linear(k, p))
     
    def forward(self, x, prev_codes):
        self.n = x.shape[0]
        self.code = self.bottom(x)
        self.z = self.demap(torch.cat([self.code, prev_codes],1)) # not saving self.z does not speed it up
        return self.output(self.z), self.code
        
    def output_code_deriv(self, to_sample, device):
        if to_sample :
            gridsize = self.n * 2
            code_grid = torch.linspace(min(self.code).item(), max(self.code).item(), steps=gridsize, dtype=torch.float, 
                                       device=device, requires_grad=False).view(gridsize, 1)
            z=self.demap(code_grid)
        else :
            z=self.z # computing this derivative ourselves is faster than using the autograd
        
#        print(z.shape)
#        print(self.demap[0].weight.shape)
        tmp = (1-z.pow(2)) * self.demap[0].weight[:,0]
        return (torch.mm(tmp, self.output[0].weight.t()))


class Model_Old (nn.Module):
    def __init__(self, k, p):
        super().__init__()
        self.bottom = nn.Sequential(nn.Linear(p, k), nn.Tanh(), nn.Linear(k, 1))
        #self.top = nn.Sequential(nn.Linear(1, k), nn.Tanh(), nn.Linear(k, p))
        # top is split into demap and output
        self.demap  = nn.Sequential(nn.Linear(1, k), nn.Tanh())
        self.output = nn.Sequential(nn.Linear(k, p))
     
    def forward(self, x):
        self.n = x.shape[0]
        self.code = self.bottom(x)
        self.z = self.demap(self.code) # not saving self.z does not speed it up
        return self.output(self.z), self.code
    
    def output_code_deriv(self, to_sample, device):
        if to_sample :
            gridsize = self.n * 2
            code_grid = torch.linspace(min(self.code).item(), max(self.code).item(), steps=gridsize, dtype=torch.float, 
                                       device=device, requires_grad=False).view(gridsize, 1)
            z=self.demap(code_grid)
        else :
            z=self.z # computing this derivative ourselves is faster than using the autograd
        
#        print(z.shape)
#        print(self.demap[0].weight.shape)
        tmp = (1-z.pow(2)) * self.demap[0].weight[:,0]
        return (torch.mm(tmp, self.output[0].weight.t()))


        
#    # make file name
#    if opt_constr=="none":
#        tmp=""
#    elif opt_constr=="constrained": 
#        tmp="_constrained"
#    else:
#        tmp = "_"+ opt_constr + str(opt_tuneParam)
#    if opt_ortho>0: 
#        tmp = tmp+ "_ortho" + str(opt_ortho)
#    filename="/"+opt_datName+"_"+opt_model+"_m"+str(opt_numCode) +"_k"+str(opt_k) + tmp +"_"+opt_str+"_"+opt_earlystop
#    filename="res/" +opt_datName+filename 
#    if opt_seed!=0: filename = filename+ "_seed" + str(opt_seed)
#    if opt_penfun!="sum": filename = filename+ "_" + str(opt_penfun)
#        
#    # create folders if needed
#    if not(os.path.isdir("res")): os.mkdir("res")  
#    if not(os.path.isdir("res/"+opt_datName)): os.mkdir("res/"+opt_datName)  
#        
#    # print and save losses
#    losses =    [out[i][0] for i in range(len(out))]
#    penlosses = [out[i][1] for i in range(len(out))]
#    l=np.column_stack((losses, penlosses))
#    print("loss pen_loss: ")  
#    print(l)  
#    np.savetxt(filename+"_losses.csv", l, delimiter=", ", fmt='%s')    
#    
#    # save codes
#    np.savetxt(filename+"_codes.csv", allcodes.cpu(), delimiter=", ", fmt='%s')            
#    
#    # save history in the first stage only
#    if opt_saveHistory=="yes":    
#        np.savetxt(filename+"_history.csv", out[0][6], delimiter=", ", fmt='%s')            
#    
#    if opt_saveLossesOnly!="yes":    
#        # save reconstructed output and weights
#        for j in range(opt_numCode):
#            np.savetxt(filename+"_output_stage"   +str(j+1)+".csv", out[j][2].cpu(), delimiter=", ", fmt='%s')    
#        #    np.savetxt(opt_datName+"/"+filename+"_decoderw_comp" +str(j+1)+".csv", out[j][4], delimiter=", ", fmt='%s')    
#        #    np.savetxt(opt_datName+"/"+filename+"_decoderb_comp" +str(j+1)+".csv", out[j][5], delimiter=", ", fmt='%s')    
#
#
#
#import argparse
#
#parser = argparse.ArgumentParser(description='AE')
#parser.add_argument('-datName', type=str, default='sim9scaled1', help='data file')
#parser.add_argument('-model', type=str, default='n', help='Model') # o for old (sequential DAM), n for new (FS-DAM), oc nc
#parser.add_argument('-numCode', type=int, default=2, help='Number of latent variables')
#parser.add_argument('-k', type=int, default=500, help='Number of nodes in the mapping layer')
#parser.add_argument('-tuneParam', type=float, default=0.01, help='Tuning parameter')
## difference between newpenalization and penalization is that in the former, the first layer penalty is calcuated on a grid instead of observed values
#parser.add_argument('-constr', type=str, default='none', help='none constrained penalization newpenalization gentle gerber hessian') 
#parser.add_argument('-ortho', type=float, default=0.01, help='Tuning parameter for orthogonality. If not, no orthogonality is enforced')
#parser.add_argument('-earlystop', type=str, default='noearlyst', help='earlystop or noearlyst, whether to use reltol to stop')
#parser.add_argument('-gpu', type=int, default=-1, help='gpu index')
#parser.add_argument('-nEpochs', type=int, default=1000, help='number of epochs to train for') # 50000 production
#parser.add_argument('-saveLossesOnly', type=str, default='no', help='yes/no. If yes, only save losses. useful in MC studies') 
#parser.add_argument('-saveHistory', type=str, default='yes', help='yes/no. If yes, save loss.item() every 100 epochs') 
#parser.add_argument('-folder', type=str, default='tmp', help='in MC studies, folder to save losses') 
#parser.add_argument('-seed', type=int, default='0', help='seed') 
#parser.add_argument('-penfun', type=str, default='sum', help='seed')  # mean is better, but we started with sum
#opt = parser.parse_args()
#
## limits to 1 cpu, otherwise it seems to be substantially slower
## for it to work, it has to come before import torch or import numpy
#if opt_gpu < 0: os.environ["OMP_NUM_THREADS"] = "1" 
#
#if __name__ == '__main__':
#    t1=datetime.datetime.now()
#    main()
#    print(datetime.datetime.now()-t1)

#    # read data
#    if(os.name=="nt") : os.chdir("D:\gdrive\DeepLearning\code") 
#    dat = pandas.read_csv("data/"+opt_datName+".csv")
#    print(dat)

## test
#model = Model_New(opt_k, p, stage)
#model = model.to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#y=x # y is output    
#loss_old = np.inf
#cnt=0
#print("iter", " reconstruct_loss", " penalized loss")
#y_pred, code = model(x, prev_codes)
#loss = loss_fn(y_pred, y)
#d = model.output_code_deriv()
#d.shape    
