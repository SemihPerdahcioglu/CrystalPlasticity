import numpy as np
import matplotlib.pyplot as plt

import CP
import utils

class deform:

    def __init__(self,grain,deform):

        self.grain=grain
        self.tot_F=np.array(deform["F"])
        self.no_inc=deform["no_inc"]

        np.set_printoptions(linewidth=200)

    def __get_F(self,no_inc,F,theta):

        F_tot=np.zeros([no_inc+1,3,3])
        F_tot[0,:,:]=np.eye(3)

        for i in range(len(F_tot)-1):

            f=(i+1)/no_inc*np.array([[ F[0], F[1] , 0.0],
                                     [ F[2], F[3] , 0.0],
                                     [  0.0,  0.0 , 0.0]])+np.eye(3)

            c=np.cos(theta*(i+1)/no_inc)
            s=np.sin(theta*(i+1)/no_inc)

            r=np.array([[   c,  -s, 0.0 ],
                        [   s,   c, 0.0 ],
                        [ 0.0, 0.0, 1.0 ]])

            F_tot[i+1,:,:]=np.matmul(r,f)

        return F_tot

    def __get_large_strains(self,F,F0,R0):

        I=np.eye(3)

        dF=np.matmul(F,np.linalg.inv(F0))

        dF_mid=0.5*(dF+I)
        h_mid=2.0*(I-np.linalg.inv(dF_mid))
        
        D_mid=0.5*(h_mid+h_mid.T)
        W_mid=0.5*(h_mid-h_mid.T)

        dR=np.matmul(np.linalg.inv(I-0.5*W_mid),I+0.5*W_mid)
        dR_mid=np.matmul(np.linalg.inv(I-0.25*W_mid),(I+0.25*W_mid))

        dR_mid_end=np.matmul(dR,np.linalg.inv(dR_mid))
        D_mid_cor=np.matmul(dR_mid_end,np.matmul(D_mid,dR_mid_end.T))

        R_mid=np.matmul(dR_mid,R0)
        R=np.matmul(dR,R0)

        return [D_mid_cor,W_mid,dR,R]

    def __print_log(self,i,state,D,E):

        print(f'Increment {i}')
        print(f'D = {D.flatten()[[0,4,8,1,5,2]]}')
        print(f'E = {E.flatten()[[0,4,8,1,5,2]]}')
        print(f's = {utils.demandel(state[0]).flatten()[[0,4,8,1,5,2]]}')
        print(f'gamma = {state[1].flatten()}')
        print(f'R_d = {state[2].flatten()}')
        print(f'Residual = {state[4]}')
        print('-------------------------------')

    def run(self):

        F=self.__get_F(self.no_inc,self.tot_F,0.0)
        E=np.zeros([self.no_inc+1,3,3])
        s=np.zeros([self.no_inc+1,3,3])
        R_d=np.zeros([self.no_inc+1,3,3])

        R_d[0,:,:]=np.eye(3)
        gamma=np.zeros([self.no_inc+1,self.grain.x_size])
        R=np.eye(3)
        Res=[0.0]

        for i in range(self.no_inc):

            [D,W,dR,R]=self.__get_large_strains(F[i+1,:,:],F[i,:,:],R)

            self.grain.set_Rd(R_d[i,:,:])
            self.grain.set_gamma(gamma[i,:])
            
            sig=utils.rotate_stress(dR,utils.mandel(s[i,:,:]))
            update=self.grain.update_stress(sig,D,W)

            s[i+1,:,:]=utils.demandel(update[0])
            gamma[i+1,:]=update[1]
            R_d[i+1,:,:]=update[2]
            C=update[3]
            Res.append(update[4])

            E[i+1,:,:]=E[i,:,:]+D

            self.__print_log(i+1,update,D,E[i+1,:,:])

        self.s=s
        self.gamma=gamma
        self.R_d=R_d
        self.Res=Res
        self.E=E

    def plot(self):

        fig, axs = plt.subplots(2,3,gridspec_kw=dict(hspace=0.5,wspace=0.5))

        strain=[s.flatten()[[0,4,8,1,5,2]] for s in self.E]
        stress=[s.flatten()[[0,4,8,1,5,2]] for s in self.s]
        rot=[s.flatten() for s in self.R_d]

        axs[0,0].plot(strain,stress)
        axs[0,0].set_xlabel('strain')
        axs[0,0].set_ylabel('stress')
        
        axs[0,1].plot(stress)
        axs[0,1].set_xlabel('increment')
        axs[0,1].set_ylabel('stress')
        
        axs[0,2].plot(strain)
        axs[0,2].set_xlabel('increment')
        axs[0,2].set_ylabel('strain')
        
        axs[1,0].plot(self.gamma)
        axs[1,0].set_xlabel('increment')
        axs[1,0].set_ylabel('gamma')

        axs[1,1].plot(rot)
        axs[1,1].set_xlabel('increment')
        axs[1,1].set_ylabel('R components')

        axs[1,2].plot(self.Res)
        axs[1,2].set_xlabel('increment')
        axs[1,2].set_ylabel('residual')

        plt.show(block=False)