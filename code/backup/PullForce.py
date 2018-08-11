
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

    
if __name__ == "__main__":
#epsilon = 1e-6;
    # parameters described in notes
    alpha = 1;
    E = 1/np.pi;
    k = 1;
    
    F = np.array([1,0]); # force applied at one end
    
    # length of vine
    l = 1;
    
    # direction of the applied force, 
    # it is the same as the direction of the last np.piece of vine
    tn = F/np.linalg.norm(F);
    
    # coordinates. the first is the fixed point, the others are all pegs
    filename = (sys.argv[1])
    #print(filename)
    ''' 1. traditional method '''
    f = open(filename, 'r') # open file
    lines = f.readlines()   # read lines into a list
    
    plist = np.array([]).reshape(0,2)
    pxy = [line.split() for line in lines] # split line into tokens
    
    pegs = np.array([]).reshape(0,2)
    for p in pxy:
        pegs = np.append(pegs,[[float(p[0]),float(p[1])]], axis = 0)
    
    #pegs = [1 1;2 2;3 1;4 2;5 1]; 
    #load pegs.txt
    
    # Fr(i,1,:} and Fr[i,2,:] are F_r^{ij}
    # e.g. Fr(3,1,:) is F_r^{32}, Fr(3,2,:) is F_r^{43}
    Fr = np.zeros((len(pegs),2,2))
    
    
    # Fr[i,:] is F_r^{i}
    Frr = np.zeros((len(pegs),2))
    for i in range(len(pegs)):
        if i != 0:
            # t1 and t2 are the vectors of two adjacent pegs
            if i == len(pegs)-1:
                t2 = tn;
            else:
                t2 = pegs[i+1,:]-pegs[i,:];
            
            t1 = pegs[i-1,:]-pegs[i,:];
            # angle
            phi = np.arccos(np.dot(t1,t2)/np.linalg.norm(t1)/np.linalg.norm(t2));
            # r1 and r2 are normal force directions
            r1 = np.array([t1[1],-t1[0]]);
            if np.dot(np.cross([(t1+r1)[0],(t1+r1)[1],0],[t1[0],t1[1],0]),np.cross([t2[0],t2[1],0],[t1[0],t1[1],0]))<0:
                r1 = np.array([-t1[1],t1[0]]);   
            
            r1 = r1/np.linalg.norm(r1);
            Fr[i-1,1,:] = alpha*E*(np.pi-phi)/np.linalg.norm(t1)*r1;
            
            r2 = np.array([t2[1],-t2[0]]);
            if np.dot(np.cross([(t2+r2)[0],(t2+r2)[1],0],[t2[0],t2[1],0]),np.cross([t1[0],t1[1],0],[t2[0],t2[1],0]))<0:
                r2 = np.array([-t2[1],t2[0]]);  
            
            r2 = r2/np.linalg.norm(r2);
            if i != len(pegs)-1:
                Fr[i+1,0,:] = alpha*E*(np.pi-phi)/np.linalg.norm(t2)*r2;
            
    
    for i in range(len(pegs)):
        if i!=0:
            Frr[i,:] = Fr[i,0,:]+Fr[i,1,:];
        
    
    
    # Ft(i,1,:} and Ft[i,2,:] are F_t^{ij}
    # e.g. Ft(3,1,:) is F_t^{32}, Ft(3,2,:) is F_t^{43}
    Ft = np.zeros((len(pegs),2,2));
    
    # Ftt[i,:] is F_t^{i}, the friction
    Ftt = np.zeros((len(pegs),2));
    
    for i in range(len(pegs)-1,0,-1):
        if i == len(pegs)-1:
            Ft[i,1,:]=F;
            t2 = F/np.linalg.norm(F);
        else:
            Ft[i,1,:]=Ft[i+1,0,:];
            t2 = (pegs[i+1,:]-pegs[i,:])/np.linalg.norm(pegs[i+1,:]-pegs[i,:]);
        
        t1 = (pegs[i-1,:]-pegs[i,:])/np.linalg.norm(pegs[i-1,:]-pegs[i,:]);
        Ftt[i,:] = k*np.linalg.norm(Frr[i,:]-np.linalg.norm((Ft[i,1,:]))*(t2+t1))*t1;
        Ft[i,0,:]=np.linalg.norm((Ft[i,1,:]))*t1-Ftt[i,:];
    
    
    # display    
    
    for i in range(len(pegs)):
        print('(%f,%f) (%f,%f)\n' %  (Ft[i,0,0],Ft[i,0,1],Ft[i,1,0],Ft[i,1,1]));
    
    for i in range(len(pegs)):
        print('(%f,%f)\n' %  (Ftt[i,0],Ftt[i,1]));
    
    for i in range(len(pegs)):
        print('(%f,%f) (%f,%f)\n' %  (Fr[i,0,0],Fr[i,0,1],Fr[i,1,0],Fr[i,1,1]));
    
    for i in range(len(pegs)):
        print('(%f,%f)\n' %  (Frr[i,0],Frr[i,1]));
    
    # plot

    plt.plot(pegs[:, 0], pegs[:, 1], 'go')
    Ft_mag = np.array([])
    for i in range(len(pegs)):
        if i==0:
            Ft_mag = np.append(Ft_mag,np.linalg.norm(Ft[1,0,:]))
        else:
            Ft_mag = np.append(Ft_mag,np.linalg.norm(Ft[i,1,:]))
    TMax = max(Ft_mag)
    TMin = min(Ft_mag)
    
    Ft_color = np.array([])
    for i in range(len(pegs)):
        Ft_color = np.append(Ft_color,int((Ft_mag[i]-TMin)/(TMax-TMin)*100))
    
    for i in range(len(pegs)):
        if i == len(pegs)-1:
            lp = pegs[-1,:]+l*tn
        else:
            lp = pegs[i+1,:]
        plt.plot(np.linspace(pegs[i,0],lp[0]),np.linspace(pegs[i,1],lp[1]),linewidth=4) # color=plt.cm.RdYlBu(Ft_color[i])
        textstr = "$F_{tension}$ = %.2f" % Ft_mag[i]
        plt.text((pegs[i,0]+lp[0])*0.5,(pegs[i,1]+lp[1])*0.5,textstr, horizontalalignment='center',verticalalignment='center',fontsize=12)
    plt.show()