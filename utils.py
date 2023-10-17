import numpy as np

def get_elastic_iso(el):

    E=el[0]
    v=el[2]

    C=np.zeros([6,6])
    D=np.zeros([6,6])

    k=E/((1.0+v)*(1.0-2.0*v))

    C[0:3,0:3]=k*np.array([[1.0-v,v,v],
                           [v,1.0-v,v],
                           [v,v,1.0-v]])
    C[3:6,3:6]=np.eye(3)*k*(1.0-2.0*v)

    D[0:3,0:3]=1.0/E*np.array([[1.0,-v,-v],
                               [-v,1.0,-v],
                               [-v,-v,1.0]])
    D[3:6,3:6]=1.0/E*np.eye(3)*(1.0+v)

    return [C,D]

def get_elastic_ortho(el,euler):

    C11=el[0]
    C12=el[1]
    C44=el[2]/2.0

    C=np.zeros([6,6])
    D=np.zeros([6,6])

    C[0:3,0:3]=C12+np.eye(3)*(C11-C12)
    C[3:6,3:6]=np.eye(3)*C44

    D[0:3,0:3]=-C12+np.eye(3)*(C11+2.0*C12)
    D[0:3,0:3]/=C11**2.0+C11*C12-2.0*C12**2.0
    D[3:6,3:6]=np.eye(3)*1.0/C44

    return [C,D]

def rotate_elastic(CD,R):

    R4=get_R4(R)

    C=np.matmul(R4,CD[0],R4.T)
    D=np.matmul(R4.T,CD[1],R4)

    return [C,D]

def fcc_planes():

    s=np.array([[-1.0, 1.0, 0.0],
                [-1.0, 0.0, 1.0],
                [ 0.0,-1.0, 1.0],
                [ 1.0, 1.0, 0.0],
                [ 1.0, 0.0, 1.0],
                [ 0.0,-1.0, 1.0],
                [ 1.0, 1.0, 0.0],
                [-1.0, 0.0, 1.0],
                [ 0.0, 1.0, 1.0],
                [-1.0, 1.0, 0.0],
                [ 1.0, 0.0, 1.0],
                [ 0.0, 1.0, 1.0]])/np.sqrt(2.0)

    m=np.array([[ 1.0, 1.0, 1.0],
                [ 1.0, 1.0, 1.0],
                [ 1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
                [ 1.0,-1.0, 1.0],
                [ 1.0,-1.0, 1.0],
                [ 1.0,-1.0, 1.0],
                [-1.0,-1.0, 1.0],
                [-1.0,-1.0, 1.0],
                [-1.0,-1.0, 1.0]])/np.sqrt(3.0)

    return [s,m]

def rotate_planes(R,sm):

    s=np.zeros(sm[0].shape)
    m=np.zeros(sm[1].shape)

    for i, v in enumerate(sm[0]):
        s[i]=np.matmul(R,v)

    for i, v in enumerate(sm[1]):
        m[i]=np.matmul(v,R.T)

    return [s,m]

def get_schmid(s,m):

    x_size=len(s)

    P=np.zeros([6,2*x_size])
    Q=np.zeros([3,3,2*x_size])

    for i in range(x_size):
        V=np.outer(s[i],m[i])

        P[:,i]=mandel(0.5*(V+V.T))
        P[:,i+x_size]=-1.0*P[:,i]
        Q[:,:,i]=0.5*(V-V.T)
        Q[:,:,i+x_size]=-1.0*Q[:,:,i]

    return [P,Q]

def mandel(a):

    c=np.sqrt(2.0)
    m=np.array([ a[0,0], a[1,1], a[2,2], c*a[0,1], c*a[1,2], c*a[0,2] ])

    return m

def demandel(a):

    c=1.0/np.sqrt(2.0)
    m=np.array([[   a[0], c*a[3], c*a[5] ],
                [ c*a[3],   a[1], c*a[4] ],
                [ c*a[5], c*a[4],   a[2] ]])

    return m

def get_R(euler):

    z2=euler[0]
    x=euler[1]
    z1=euler[2]

    c_z1=np.cos(z1)
    s_z1=np.sin(z1)
    c_x=np.cos(x)
    s_x=np.sin(x)
    c_z2=np.cos(z2)
    s_z2=np.sin(z2)

    R=np.array([[ c_z1*c_z2-s_z1*s_z2*c_x,  s_z1*c_z2+c_z1*s_z2*c_x,  s_z2*s_x],
                [-c_z1*s_z2-s_z1*c_z2*c_x, -s_z1*s_z2+c_z1*c_z2*c_x,  c_z2*s_x],
                [                s_z1*s_x,                -c_z1*s_x,       c_x]])

    return R

def get_R4(R):

    R4_11=R*R
    
    R4_12=np.sqrt(2)*np.array([[ R[0,1]*R[0,2], R[0,0]*R[0,2], R[0,0]*R[0,1] ],
                               [ R[1,1]*R[1,2], R[1,0]*R[1,2], R[1,0]*R[1,1] ],
                               [ R[2,1]*R[2,2], R[2,0]*R[2,2], R[2,0]*R[2,1] ]])
    
    R4_21=np.sqrt(2)*np.array([[ R[1,0]*R[2,0], R[1,1]*R[2,1], R[1,2]*R[2,2] ],
                               [ R[0,0]*R[2,0], R[0,1]*R[2,1], R[0,2]*R[2,2] ],
                               [ R[0,0]*R[1,0], R[0,1]*R[1,1], R[0,2]*R[1,2] ]])
    
    R4_22=np.array([[ R[1,1]*R[2,2]+R[1,2]*R[2,1], R[1,0]*R[2,2]+R[1,2]*R[2,0], R[1,0]*R[2,1]+R[1,1]*R[2,0] ],
                    [ R[0,1]*R[2,2]+R[0,2]*R[2,1], R[0,0]*R[2,2]+R[0,2]*R[2,0], R[0,0]*R[2,1]+R[0,1]*R[2,0] ],
                    [ R[0,1]*R[1,2]+R[0,2]*R[1,1], R[0,0]*R[1,2]+R[0,2]*R[1,0], R[0,0]*R[1,1]+R[0,1]*R[1,0] ]])

    R4=np.hstack((np.vstack((R4_11,R4_21)),np.vstack((R4_12,R4_22))))

    return R4

def get_dRp(Q,lando,W):
    
    Wp=np.zeros([3,3])
    for i,f in enumerate(lando):
        Wp+=f*Q[:,:,i]

    return get_dR(W-Wp)

def get_dR(W):

    I=np.eye(3)

    return np.matmul(np.linalg.inv(I-0.5*W),(I+0.5*W))

def rotate_stress(R,s):
    
    return mandel(np.matmul(R,np.matmul(demandel(s),R.T)))

def get_flowstress(el,hard,gamma):

    x_size=len(gamma)

    G=el[0]/(2.0*(1.0+el[2])) if el[1]<1.0e-8 else np.sqrt(0.5*el[2]*(el[0]-el[1]))
    
    rho=np.zeros(x_size)
    tau_f=np.zeros(2*x_size)

    Kd=get_interaction_fcc()

    for i, g in enumerate(gamma):
        rho[i]=hard[2]*(1.0-(1.0-hard[1]/hard[2])*np.exp(-g/hard[3]))

    alpha=np.matmul(Kd,rho)

    tau_f[0:x_size]=hard[4]+G*hard[0]*np.sqrt(alpha)
    tau_f[x_size:2*x_size]=tau_f[0:x_size]

    dg_quart=np.zeros([x_size,x_size])
    for i in range(x_size):
        for j in range(x_size):
            dg_quart[i,j]=G*hard[0]*Kd[i,j]*(hard[2]-rho[j])/hard[3]/(2.0*np.sqrt(alpha[i]))

    dg=np.hstack((np.vstack((dg_quart,dg_quart)),np.vstack((dg_quart,dg_quart))))

    return [tau_f,dg]

def get_interaction_fcc():

    G=np.array([0.122,0.122,0.07,0.625,0.137,0.122])

    Kd=np.array([[ G[0], G[1], G[1], G[2], G[5], G[4], G[2], G[4], G[5], G[3], G[4], G[4] ],
                 [ G[1], G[0], G[1], G[5], G[2], G[4], G[4], G[3], G[4], G[4], G[2], G[5] ],
                 [ G[1], G[1], G[0], G[4], G[4], G[3], G[5], G[4], G[2], G[4], G[5], G[2] ],
                 [ G[2], G[5], G[4], G[0], G[1], G[1], G[3], G[4], G[4], G[2], G[4], G[5] ],
                 [ G[5], G[2], G[4], G[1], G[0], G[1], G[4], G[2], G[5], G[4], G[3], G[4] ],
                 [ G[4], G[4], G[3], G[1], G[1], G[0], G[4], G[5], G[2], G[5], G[4], G[2] ],
                 [ G[2], G[4], G[5], G[3], G[4], G[4], G[0], G[1], G[1], G[2], G[5], G[4] ],
                 [ G[4], G[3], G[4], G[4], G[2], G[5], G[1], G[0], G[1], G[5], G[2], G[4] ],
                 [ G[5], G[4], G[2], G[4], G[5], G[2], G[1], G[1], G[0], G[4], G[4], G[3] ],
                 [ G[3], G[4], G[4], G[2], G[4], G[5], G[2], G[5], G[4], G[0], G[1], G[1] ],
                 [ G[4], G[2], G[5], G[4], G[3], G[4], G[5], G[2], G[4], G[1], G[0], G[1] ],
                 [ G[4], G[5], G[2], G[5], G[4], G[2], G[4], G[4], G[3], G[1], G[1], G[0] ]])

    return Kd