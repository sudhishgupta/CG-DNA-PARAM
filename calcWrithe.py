import warnings
warnings.filterwarnings('ignore')
import numpy as np
import MDAnalysis
import os
import time

def Read(u, trajectory=None):
    
    total_res = len(u.residues)
    nbp = total_res // 2
    
    strandA = []
    strandB = []

    for fi, ts in enumerate(u.trajectory if trajectory is not None else [u.trajectory.ts]):
        frameA = []
        frameB = []
        for i,res in enumerate(range(total_res-1)):
            u_atoms = u.residues
            base = u_atoms[res]
            resname = u.residues[res].resname
            #print(resname, end=',')
            xprm = base.atoms.select_atoms("name BB2")
            xprm_cog = xprm.center_of_geometry()
            #print(xprm_cog.shape)
            if i+1<=300:
                frameA.append(xprm_cog)
            else:
                frameB.append(xprm_cog)

        frameB_rev = frameB[::-1]

        strandA.append(frameA)
        strandB.append(frameB_rev)
        
        E = xprm.positions.copy()
        # print(f'res--{i+1} :: {xprm_cog}')

           
    #r = [0.25*(strandA[j] + strandA[(j+1)%nbp] + strandB_rev[j] + strandB_rev[(j+1)%nbp]) for j in range(nbp)]

    rA = np.transpose(np.array(strandA),  (2, 0, 1))
    rB = np.transpose(np.array(strandB),  (2, 0, 1))

    r = np.zeros_like(rA)

    for j in range(nbp):
        r[:, :, j] = 0.25*( rA[:, :, j] + rA[:, :, (j+1)%nbp] + rB[:, :, j] + rB[:, :, (j+1)%nbp] )
    
    return rA, rB, r
    

def haxis(nbp, nstep, r, rA):
    rC = np.zeros_like(rA)
    for j in range(0, nbp):
        if j%100 ==0:
            print('now working on basepairs '+str(j)+'s....')
        
        SUM = np.zeros((3, nstep))
        
        for t in range(0, nstep):
            SUM[:, t] += r[:,t,j]
            k=0
            while k<5:
                k+=1
                # print(k,(r[:,t,(j-k)%nbp] + r[:,t,(j+k)%nbp]).shape)
                # print(SUM.shape)
                SUM[:,t] += r[:,t,(j-k)%nbp] + r[:,t,(j+k)%nbp]

            rC[:,t,j] = SUM[:,t]/(2*k+1)
    return rC
        
def Arctan360(x,y):
    # find arctan of y/x and return values between -180 to 180
    if x >= 0 and y >= 0: #Q1
        A = np.arctan( y/x )
    elif x < 0 and y >= 0: #Q2
        A = np.arctan( y/x ) + np.pi
    elif x < 0 and y < 0: #Q3
        A = np.arctan( y/x ) - np.pi
    else: #Q4
        A = np.arctan( y/x )
    return A
    
def setX(V):
    V = V / np.linalg.norm(V)
    x = V[0]
    y = V[1]
    z = V[2]
    # defining rotation matrix about z axis and euler angles as: 
    # Rz(C)
    C = -Arctan360(x,y)
    Rz = np.array([[np.cos(C), -np.sin(C), 0.0], [np.sin(C), np.cos(C), 0.0], [0.0, 0.0, 1.0]])
    return Rz


def setZ(V):
    V = V / np.linalg.norm(V)
    x = V[0]
    y = V[1]
    z = V[2]
    # defining rotation matrix about x,y axis and euler angles as: 
    # Rxy(A,B) = dot( Ry(B),Rx(A) )
    # Given r = (x,y,z) and uz = (0,0,1)
    # solve the equation uz = dot( Rxy, r )
    A = Arctan360( z,y )
    B = np.arctan( -x / np.sqrt(y**2+z**2) )
    # calculating Rxy
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(A), -np.sin(A)], [0.0, np.sin(A), np.cos(A)]])
    Ry = np.array([[np.cos(B), 0.0, np.sin(B)], [0.0, 1.0, 0.0], [-np.sin(B), 0.0, np.cos(B)]])
    Rxy = np.dot( Ry, Rx )
    return Rxy

def Cross(A, B):
    C = np.zeros( np.shape(A) )
    C[0] = A[1]*B[2] - A[2]*B[1]
    C[1] = A[2]*B[0] - A[0]*B[2]
    C[2] = A[0]*B[1] - A[1]*B[0]
    return C
def Dot(A, B):
    D = np.zeros( np.shape(A[0]) )
    D = A[0]*B[0] + A[1]*B[1] + A[2]*B[2]
    return D
# vector's magnitude
def Size( A ):
    s = np.sqrt( A[0]**2 + A[1]**2 + A[2]**2 )
    return s
    
def sinreg(nbp, nstep, r, r1):
    SinReg = np.zeros((nstep,nbp+1))

    for j in range(0, nbp):
        if j%100 ==0 :
            print('now working on basepairs '+str(j)+'s....')

        m = np.linspace(j-1, j+1, num=3)%nbp
        #print(m)
        v0 = r1[:,:,int(m[1])] - r1[:,:,int(m[0])]
        v1 = r1[:,:,int(m[2])] - r1[:,:,int(m[1])]

        C = Cross( v1, v0 )
        M = r[:,:,int(m[0])] - r1[:,:,int(m[0])]

        SinReg[:,j+1] = Size(Cross(M,C)) / (Size(M)*Size(C))

        for i in range(0, nstep):
            if Dot(v1-v0, M)[i] < 0:
                SinReg[i,j+1] = -SinReg[i,j+1]

        SinReg[:,0] = np.linspace(0.01, 0.01*nstep, num=nstep)

        np.savetxt('sinreg.ser', SinReg, fmt='%8.3f')

    return SinReg
    
    
def twist( P11, P12, P21, P22, Z ):
    # Z is the vector defining Z axis
    # the two vectors
    r1 = P12 - P11
    r2 = P22 - P21
    # normalization
    r1 = r1 / np.linalg.norm(r1)
    r2 = r2 / np.linalg.norm(r2)
    Z = Z / np.linalg.norm(Z)
    # rotate the vector Z to z axis
    Rxy = setZ(Z)
    # rotate r1 and r2 along with Z using rotation matrix Rxy 
    r1 = np.dot( Rxy,r1 )
    r2 = np.dot( Rxy,r2 )
    # rotate r1 about z axis so that y becomes 0 and rotate r2 along
    Rz = setX(r1)
    r2 = np.dot( Rz,r2 )
    # now calculatin the twist
    return Arctan360(r2[0],r2[1])*180.0/np.pi

    
def Twist(nbp, nstep, rA, rB, rC):
    tw = np.zeros((nstep, nbp))

    for j in range(0, nbp):
        if j%100 == 0:
            print('now working on basepairs '+str(j)+'s....')
        
        for t in range(0, nstep):
            Z = rC[:,t,(j+1)%nbp] - rC[:,t,(j-1)%nbp]
        
            tw[t,j] = twist(rA[:,t,j], rB[:,t,j], rA[:,t,(j+1)%nbp], rB[:,t,(j+1)%nbp], Z)
    return tw

def caxis(nbp, nstep, r, tw):
    r1 = np.zeros(np.shape(r))

    for j in range(0, nbp):
        if j%100 == 0:
            print('now working on basepairs '+str(j)+'s....')
        Tw = np.zeros(nstep)
        SUM = np.zeros((3, nstep))

        for t in range(nstep):
            Tw[t] += tw[t,j]
            SUM[:,t] += r[:,t,j]

            k=0

            while (Tw[t] < 360.0):
                k+=1
                prev = Tw[t]
                Tw[t] += tw[t,(j-k)%nbp] + tw[t,(j+k)%nbp]
                SUM[:,t] += r[:,t,(j-k)%nbp] + r[:,t,(j+k)%nbp]

            w = (360.0 - prev) / (Tw[t] - prev)

            SUM[:,t] = SUM[:,t] - (1-w) * (r[:,t,(j-k)%nbp] + r[:,t,(j+k)%nbp])

            W = np.array([w, w, w])

            r1[:,t,j] = SUM[:,t]/(2*(k+W)-1)

    return r1

def MakeFiles( name, nbp, nstep, r, r1 ):

    os.makedirs(name, exist_ok=True)
    
    fxyz = open(f'{name}/C.xyz', 'w')
    fxyz1 = open(f'{name}/C1.xyz', 'w')

    for i in range(0, nstep):
        if i % 1000 == 0:
            print(f'{i} steps have been written for .xyz output')

        fxyz.write(f'{nbp}\n\n')
        fxyz1.write(f'{nbp}\n\n')

        for j in range(0, nbp):
            fxyz.write("H %8.3f %8.3f %8.3f \n"%(r[0][i][j], r[1][i][j], r[2][i][j]))
            fxyz1.write("H %8.3f %8.3f %8.3f \n"%(r1[0][i][j], r1[1][i][j], r1[2][i][j]))



    fc = open(f'{name}/C.3col','w')
    fc1 = open(f'{name}/C1.3col', 'w')

    for i in range(0, nstep):

        if i%1000 == 0:
            print(f'{i} steps have been written for .3col output')

        for j in range(0, nbp):
            fc.write("%8.3f %8.3f %8.3f \n"%(r[0][i][j], r[1][i][j], r[2][i][j]))
            fc1.write("%8.3f %8.3f %8.3f \n"%(r1[0][i][j], r1[1][i][j], r1[2][i][j]))


    fc.close()
    fc1.close()



# read the 3col coordinate file and split it by the timestep
def readf( filename, nbp, nstep ):
    whole = np.loadtxt( filename )
    x = []
    t = 0
    for i in range(0, nstep):
        x.append( whole[t*nbp:(t+1)*nbp,:] )
        t += 1
    return np.array( x )

# # save data
# def savevar(name, data):
#     ''' saves all the intermediate variables '''
#     #data : dict of variables to be saved
#     for key, value in data.items():
#         np.savetxt(f'{name}/{key}.txt', value)
    


# writhe for a single timestep
def wr( x, t, l, axis=2 ):
    Shape = np.shape( x[t] )
    y = np.zeros(( Shape[0]+1, Shape[1] ))
    y[:Shape[0],:] = x[t] # read the array for one bpstep
    y[Shape[0],:] = x[t,0] # add the head at the bottom
    Wr = 0
    for j in range(0,l):
        for k in range(0,j):
            tj = y[j+1] - y[j] # tangents of jth base pairs
            tk = y[k+1] - y[k] # tangents of kth base pairs
            rjk = y[j] - y[k] # vector joining j and k
            # now performing discretized Gauss Integral
            # W is an individual contribution from a pair of tangent vector
            W = np.dot( rjk, np.cross(tj, tk) ) / (np.linalg.norm(rjk))**3.0 / (2*np.pi)
            Wr += W
    return Wr

def main_wr(name, nbp, nstep):
    print('Calculating Writhe...')
    # read the file
    inp = name+'/C1.3col'
    X = readf( inp, nbp, nstep )
    l = len( X[0] )
    # calculate the writhe for nstep timesteps
    Writhe = []
    for t in range(0, nstep):
        if t%10 == 0:
            print ('now working on steps '+str(t)+'s...')
        Writhe.append( [ t+1, wr( X, t, l ) ] ) 
    Writhe = np.array( Writhe )

    np.savetxt( name+'/writhe.ser', Writhe, fmt='%5d %9.4f' )

import argparse
def get_args():
    parser = argparse.ArgumentParser(description="Run WR-CG")

    parser.add_argument(
        "--top_pdb",
        type=str,
        default=None,
        help="Topology/structure file (PDB or GRO)"
    )
    parser.add_argument(
        "--nbp",
        type=int,
        default=None,
        help="Number of BasePairs"
    )

    parser.add_argument(
        "--traj_file",
        type=str,
        default=None,
        help="Trajectory file (XTC/TRR) or 'None' for single-frame mode"
    )

    parser.add_argument(
        "--out_folder",
        type=str,
        default='CGWR_output',
        help="Output folder to store results"
    )

    return parser.parse_args()


def main():
    args = get_args()

    top_pdb = args.top_pdb
    traj_file = args.traj_file
    out_folder = args.out_folder
    nbp = int(args.nbp)

    if traj_file is not None:
        uu = MDAnalysis.Universe(top_pdb, traj_file)
        nsteps = len(uu.trajectory)
    else:
        uu = MDAnalysis.Universe(top_pdb)
        nsteps=1
    
    
    print('Reading Trajectories')
    start = time.time()
    rA,rB,r = Read(uu, trajectory=traj_file)
    end = time.time()
    print(f'Done!, Time Taken : {end-start : .2f}s')

    print('Calculating Helical Axis')
    start = time.time()
    rC = haxis(nbp, nsteps, r, rA)
    end = time.time()
    print(f'Done!, Time Taken : {end-start : .2f}s')

    print('Now Working on Topological Twist of DNA Strands')
    start = time.time()
    tw = Twist(nbp, nsteps, rA, rB, rC)
    end = time.time()
    print(f'Done!, Time Taken : {end-start : .2f}s')

    print('Doing Some More Calculations')
    start = time.time()
    r1 = caxis(nbp,nsteps,r,tw)
    end = time.time()
    print(f'Done!, Time Taken : {end-start : .2f}s')

    print('Working out SinReg Operation')
    start = time.time()
    sr = sinreg(nbp, nsteps, r, r1)
    end = time.time()
    print(f'Done!, Time Taken : {end-start : .2f}s')

    print('Writing Intermediate Files')
    start = time.time()
    MakeFiles(out_folder,nbp, nsteps, r, r1)
    end = time.time()
    print(f'Done!, Time Taken : {end-start : .2f}s')

    print(f'Calculating Writhe')
    start = time.time()
    main_wr(out_folder,nbp,nsteps)
    end = time.time()
    print(f'Done!, Time Taken : {end-start : .2f}s')

    print(f'Saving Workspace Variables')
    data = {'rA' : rA, 'rB' : rB, 'r' : r, 'rC' : rC, 'tw' : tw, 'r1' : r1, 'sr' : sr}
    
    with open(f'{out_folder}/workspace.pkl','wb') as file:
        pickle.dump(data, file)

    print('Saved !')
    

if __name__ == '__main__':
    print('='*40)
    print('Writhe Calculation for Coarse-Grain DNA Models')
    print('='*40)
    print('\n')

    main()    
