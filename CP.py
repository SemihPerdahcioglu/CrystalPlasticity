import numpy as np
import utils

class crystal:

    def __init__(self,grain):
        
        self.x_size=12
        self.el=np.array(grain["el"])
        self.hard=np.array(grain["hard"])
        self.euler=np.array(grain["euler"])
        
        self.__Rm=utils.get_R(self.euler)
        self.__Rd=np.eye(3)
        self.__gamma=np.zeros(self.x_size)

        self.__update_elastic()
        self.__update_schmid()

    def __update_schmid(self):
        
        if self.x_size==12:
            [s,m]=utils.rotate_planes(np.matmul(self.__Rd,self.__Rm),utils.fcc_planes())
        else:
            s=0.0
            m=0.0

        self.__s=s
        self.__m=m

        self.__schmid=utils.get_schmid(s,m)

    def __update_elastic(self):

        if self.el[1]>1.0e-8:
            [C,D]=utils.rotate_elastic(utils.get_elastic_ortho(self.el),np.matmul(self.__Rd,self.__Rm))
        else:
            [C,D]=utils.get_elastic_iso(self.el)

        self.__Cel=C
        self.__Del=D

    def __cur_schmid(self,dR):

        [s,m]=utils.rotate_planes(dR,[self.__s,self.__m])

        return utils.get_schmid(s,m)

    def __cur_elastic(self,dR):

        [C,D]=[self.__Cel,self.__Del]

        if self.el[1]>1.0e-8:
            [C,D]=utils.rotate_elastic([C,D],dR)

        return [C,D]

    def __get_phi(self,s,P,dgamma=0.0):

        tau_r=np.matmul(s,P)
        
        gamma=self.__gamma if np.isscalar(dgamma) else self.__gamma+dgamma[0:self.x_size]+dgamma[self.x_size:2*self.x_size]
        tau_f,dg=utils.get_flowstress(self.el,self.hard,gamma)

        return [tau_r-tau_f,dg]

    def set_Rd(self,Rd):

        self.__Rd=Rd
    
    def set_gamma(self,gamma):

        self.__gamma=gamma

    def update_stress(self,s0,D,W):

        tol=1.0e-8
        mu=1.0e-7

        self.__update_schmid()
        self.__update_elastic()

        dR=utils.get_dR(W)
        [P,Q]=self.__cur_schmid(dR)
        [Cel,Del]=self.__cur_elastic(dR)
        
        w=np.maximum(tol,np.maximum(-self.__get_phi(s0,P)[0],0.0e0))
        lando=mu/w

        s=np.copy(s0)

        ds=np.zeros(6)
        dlando=np.zeros(self.x_size*2)
        dw=np.zeros(self.x_size*2)

        i=0
        Res=1.0
        while (Res>=1.0 and i<25):
            i+=1

            if i>1:
                dlando=np.linalg.solve(dRdl,R_fi)
                dw=mu/lando-w-w/lando*dlando

                if np.min(lando+dlando)<0.0:
                    dlando*=0.95*(-lando/dlando)[lando+dlando<0.0].min()
                    dw=mu/lando-w-w/lando*dlando
                if np.min(w+dw)<0.0:
                    dw*=0.95*(-w/dw)[w+dw<0.0].min()
                    dlando=(mu/lando-w-dw)*lando/w

                ds=-np.matmul(Cel,np.matmul(P,dlando))+R_sig

            s+=ds
            lando+=dlando
            w+=dw

            dR=utils.get_dRp(Q,lando,W)
            [P,Q]=self.__cur_schmid(dR)
            [Cel,Del]=self.__cur_elastic(dR)

            ds_tr=np.matmul(Cel,utils.mandel(D))
            dep=np.matmul(P,lando)
            fi,dg=self.__get_phi(s,P,lando)

            R_sig=ds_tr-(s-s0)-np.matmul(Cel,dep)
            R_fi=fi+mu/lando+np.matmul(R_sig,P)

            dRdl=np.matmul(P.T,np.matmul(Cel,P))+dg+np.diag(w/lando)

            R_norm=np.sqrt(np.dot(R_fi,R_fi))
            Res0=R_norm if i==1 else Res0
            Res=R_norm/Res0/tol

        gamma=self.__gamma+np.maximum(lando[0:self.x_size]+lando[self.x_size:2*self.x_size],0.0)
        Rd=np.matmul(utils.get_dRp(Q,lando,W),self.__Rd)
        C=np.linalg.inv(Del+np.matmul(P,np.linalg.solve(dg+np.diag(w/lando),P.T)))

        return [s,gamma,Rd,C,Res]



