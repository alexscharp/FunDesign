
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import seaborn as sns

sns.set() # use seaborn style


def compute_force(pegs_all, pegs, ax):
    # parameters described in notes
    alpha = 1;
    E = 1/np.pi;
    k = 1;

    F = np.array([-1,1]); # force applied at one end

    # figure, ax = plt.subplots()
    # ax.set_xlim([xmin, xmax])
    # ax.set_ylim([ymin, ymax])
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])

    ax.plot(pegs_all[:, 0], pegs_all[:, 1],'go',markersize=6)

    # Fr(i,1,:} and Fr[i,2,:] are F_r^{ij}
    # e.g. Fr(3,1,:) is F_r^{32}, Fr(3,2,:) is F_r^{43}
    Fr = np.zeros((len(pegs),2,2))


    # Fr[i,:] is F_r^{i}
    Frr = np.zeros((len(pegs),2))

    sign_arr = -1*np.ones([len(pegs),1])
    sign_arr[0] = 0
    sign_arr[-1] = 0

    init_angles = np.ones([len(pegs),1])*np.pi
    init_angles[0] = 0
    init_angles[-1] = 0
    for i in range(len(pegs)):
        if i != 0 and i != len(pegs)-1:
            # t1 and t2 are the vectors of two adjacent pegs
            t2 = pegs[i+1,:]-pegs[i,:];
            t1 = pegs[i-1,:]-pegs[i,:];
            if np.linalg.norm(t2) == 0 or np.linalg.norm(t1) == 0:
                print("error input")
                sys.exit()
            # angle
            aphi = np.dot(t1,t2)/np.linalg.norm(t1)/np.linalg.norm(t2)
            if aphi>1:
                aphi = 1
            elif aphi<-1:
                aphi = -1
            phi = np.arccos(aphi);

            # r1 and r2 are normal force directions
            r1 = np.array([t1[1],-t1[0]]);
            if np.dot(np.cross([(t1+r1)[0],(t1+r1)[1],0],[t1[0],t1[1],0]),np.cross([t2[0],t2[1],0],[t1[0],t1[1],0]))>0:
                r1 = np.array([-t1[1],t1[0]]);   

            r1 = r1/np.linalg.norm(r1);
            Fr[i-1,1,:] = alpha*E*(init_angles[i]-phi)/np.linalg.norm(t1)*r1;

            r2 = np.array([t2[1],-t2[0]]);
            if np.dot(np.cross([(t2+r2)[0],(t2+r2)[1],0],[t2[0],t2[1],0]),np.cross([t1[0],t1[1],0],[t2[0],t2[1],0]))>   0:
                r2 = np.array([-t2[1],t2[0]]);  

            r2 = r2/np.linalg.norm(r2);
            Fr[i+1,0,:] = alpha*E*(init_angles[i]-phi)/np.linalg.norm(t2)*r2;

    for i in range(1,len(pegs)-1):
        if i!=0:
            Frr[i,:] = Fr[i,0,:]+Fr[i,1,:];

    # Ft(i,1,:} and Ft[i,2,:] are F_t^{ij}
    # e.g. Ft(3,1,:) is F_t^{32}, Ft(3,2,:) is F_t^{43}
    Ft = np.zeros((len(pegs),2,2));

    # normal force
    Fn = np.zeros((len(pegs),2));
    # Ftt[i,:] is F_t^{i}, the friction
    Ftt = np.zeros((len(pegs),2));

    for i in range(len(pegs)-2,0,-1):
        t2 = (pegs[i+1,:]-pegs[i,:])/np.linalg.norm(pegs[i+1,:]-pegs[i,:]);
        t1 = (pegs[i-1,:]-pegs[i,:])/np.linalg.norm(pegs[i-1,:]-pegs[i,:]);
        if i == len(pegs)-2:    
            Ft[i,1,:]=0 #k*np.linalg.norm(Frr[i,:])*t2
        else:
            Ft[i,1,:]=-Ft[i+1,0,:];

        Ftt[i,:] = -k*np.linalg.norm(Frr[i,:]+sign_arr[i]*np.linalg.norm((Ft[i,1,:]))*(t2+t1))*t1; #
        Fn[i,:] = Frr[i,:]+sign_arr[i]*np.linalg.norm((Ft[i,1,:]))*(t2+t1)#
        Ft[i,0,:]=np.linalg.norm((Ft[i,1,:]))*t1-Ftt[i,:];

    """
    for i in range(len(pegs)):
        print('Ft: (%f,%f) (%f,%f)\n' %  (Ft[i,0,0],Ft[i,0,1],Ft[i,1,0],Ft[i,1,1]));

    for i in range(len(pegs)):
        print('Ftt: (%f,%f)\n' %  (Ftt[i,0],Ftt[i,1]));

    for i in range(len(pegs)):
        print('Fr: (%f,%f) (%f,%f)\n' %  (Fr[i,0,0],Fr[i,0,1],Fr[i,1,0],Fr[i,1,1]));

    for i in range(len(pegs)):
        print('Frr: (%f,%f)\n' %  (Frr[i,0],Frr[i,1]));
    """


    Ft_mag = np.array([])
    for i in range(len(pegs)):
        if i==0:
            Ft_mag = np.append(Ft_mag,np.linalg.norm(Ft[1,0,:]))
        else:
            Ft_mag = np.append(Ft_mag,np.linalg.norm(Ft[i,1,:]))

    Fr_unit = np.zeros((len(pegs),2));
    Ft_unit = np.zeros((len(pegs),2));
    for i in range(len(pegs)):
        if np.linalg.norm(Fn[i,:])>1e-9:
            Fr_unit[i,:] = Fn[i,:]/np.linalg.norm(Fn[i,:])

        if np.linalg.norm(Ftt[i,:])>1e-9:
            Ft_unit[i,:] = Ftt[i,:]/np.linalg.norm(Ftt[i,:])


    for i in range(len(pegs)-1):
        lp = pegs[i+1,:]
        ax.plot(np.linspace(pegs[i,0],lp[0]),np.linspace(pegs[i,1],lp[1]),linewidth=1,color='b') # color=plt.cm.RdYlBu(Ft_color[i])
        textstr = "$T$ = %.2f" % Ft_mag[i]
        ofst = 0.12
        ax.text((pegs[i,0]+lp[0])*0.5-ofst,(pegs[i,1]+lp[1])*0.5,textstr, horizontalalignment='center',verticalalignment='center',fontsize=12)

        if i != 0 and i != len(pegs)-1:
            if np.linalg.norm(Fn[i, :])>1e-6:
                ax.quiver(pegs[i, 0],pegs[i, 1],Fr_unit[i, 0],Fr_unit[i, 1],width=0.003)
            if np.linalg.norm(Ftt[i, :])>1e-6:
                ax.quiver(pegs[i, 0],pegs[i, 1],Ft_unit[i, 0],Ft_unit[i, 1],color='r',width=0.003)
            scale = 0.12
            textstr = "$N$ = %.2f" % np.linalg.norm(Fn[i,:])
            ax.text((pegs[i,0]+Fr_unit[i, 0]*scale)+ofst,(pegs[i,1]+Fr_unit[i, 1]*scale),textstr, horizontalalignment='center',verticalalignment='center',fontsize=8)
            scale = 0.12
            textstr = "$f$ = %.2f" % np.linalg.norm(Ftt[i,:])
            ax.text((pegs[i,0]+Ft_unit[i, 0]*scale),(pegs[i,1]+Ft_unit[i, 1]*scale),textstr, horizontalalignment='center',verticalalignment='center',fontsize=8)


if __name__ == "__main__":
    pegs = np.array([[-0.18, -0.29], [0.25, 0.35], [0.19, 0.49],
        [-0.2, 0.63], [-0.23, 0.648], [-0.5324176, 0.34558240]])
    pegs_all = np.array([[0.25, 0.35], [0.4, 0.1],
        [0.47, 0.29], [0.38, 0.38],
        [0.6, 0.4], [0.4, 0.6], [0.20, 0.7],
        [0.20, 0.15], [0.05, 1.27],
        [0.25, 0.90], [0.03, 0.40],
        [0.58, 0.63], [0.15, 0.05], [0.31, 0.20],
        [0.64, 0.23], [0.31, 0.93], [0.03, 0.38],
        [0.30, 0.03], [0.56, 0.06], [0.13, 1.0],
        [0.38, 0.74], [0.11, 1.28], [0.17, 1.09],
        [-0.01, 1.13], [0.42, 0.49],
        [0.00, 0.88], [-0.11, 1.28], [0.45, 0.47],
        [0.12, 0.33], [-0.02, 1.31],
        [0.15, 0.69], [0.14, 0.46],
        [0.19, 0.49], [-0.23, 0.648],
        [-0.15, 0.62],
        ])
    xmin, xmax, ymin, ymax = -1.0, 1.4, -0.4, 2.0
    figure, ax = plt.subplots()
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    compute_force(pegs_all, pegs, ax)
    plt.show()
