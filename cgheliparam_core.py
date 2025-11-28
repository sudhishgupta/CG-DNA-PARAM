import numpy as np
from numpy import linalg as LA
import MDAnalysis
import os
from tqdm import tqdm

def run_cgheliparam(topology, trajectory=None, library_path='./data/', outfile_prefix=None, verbose=True):
    """
    Calculate basepair / step / local helix parameters for either:
      - a single-file Universe (topology only; trajectory=None) OR
      - a topology + trajectory (topology, trajectory)

    Returns:
      basepair_parameters_all:  (n_frames, basepairs, 6)
      step_parameters_all:      (n_frames, steps, 6)
      local_parameters_all:     (n_frames, steps, 6)

    If outfile_prefix is provided, results are saved as numpy .npz:
      {outfile_prefix}_heliparams.npz
    """

    # load Universe (supports both single-frame and trajectories)
    if trajectory is None:
        u = MDAnalysis.Universe(topology)
    else:
        u = MDAnalysis.Universe(topology, trajectory)

    # Pre-load standard CG pdbs once (reuse across frames)
    std_files = {
        'A': os.path.join(library_path, 'CG_A_std.pdb'),
        'C': os.path.join(library_path, 'CG_C_std.pdb'),
        'G': os.path.join(library_path, 'CG_G_std.pdb'),
        'T': os.path.join(library_path, 'CG_T_std.pdb'),
    }
    std_univs = {}
    for k, path in std_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Standard file not found: {path}")
        std_univs[k] = MDAnalysis.Universe(path)

    # helper rotation utils (kept same logic as your original)
    def rotMatrix(v, ang):
        return np.array([
            [np.cos(ang) + (1 - np.cos(ang)) * v[0],
             (1 - np.cos(ang)) * v[0] * v[1] - v[2] * np.sin(ang),
             (1 - np.cos(ang)) * v[0] * v[2] + v[1] * np.sin(ang)],
            [(1 - np.cos(ang)) * v[0] * v[1] + v[2] * np.sin(ang),
             np.cos(ang) + (1 - np.cos(ang)) * v[1] * v[1],
             (1 - np.cos(ang)) * v[1] * v[2] - v[0] * np.sin(ang)],
            [(1 - np.cos(ang)) * v[0] * v[2] - v[1] * np.sin(ang),
             (1 - np.cos(ang)) * v[1] * v[2] + v[0] * np.sin(ang),
             np.cos(ang) + (1 - np.cos(ang)) * v[2] * v[2]]
        ])

    def angle(v1, v2):
        # clip to [-1, 1] for numerical stability
        denom = (LA.norm(v1) * LA.norm(v2))
        if denom == 0:
            return 0.0
        val = np.dot(v1, v2) / denom
        val = np.clip(val, -1.0, 1.0)
        return np.arccos(val)

    # lsfit rewritten to reuse preloaded std_univs
    def lsfit(residue_name, residue_number, u_atoms):
        """
        u_atoms: a reference MDAnalysis AtomGroup for the current Universe (frame)
        residue_name: e.g. 'A', 'C', 'G', 'T' (or contains those chars)
        residue_number: index among residues in the Universe
        returns: o (3-vector), R (3x3)
        """
        base = u_atoms[residue_number]  # an MDAnalysis Residue (AtomGroup)
        # choose selection for this residue
        if 'A' in residue_name:
            xprm = base.atoms.select_atoms("name SC1 or name SC2 or name SC3 or name SC4")
            stnd = std_univs['A']
        elif 'C' in residue_name:
            xprm = base.atoms.select_atoms("name SC1 or name SC2 or name SC3")
            stnd = std_univs['C']
        elif 'G' in residue_name:
            xprm = base.atoms.select_atoms("name SC1 or name SC2 or name SC3 or name SC4")
            stnd = std_univs['G']
        elif 'T' in residue_name:
            xprm = base.atoms.select_atoms("name SC1 or name SC2 or name SC3")
            stnd = std_univs['T']
        else:
            raise ValueError(f"Unexpected residue name: {residue_name}")

        # centers
        # use center_of_geometry() to be robust
        stnd_cog = stnd.atoms.center_of_geometry()
        xprm_cog = xprm.center_of_geometry()

        S = stnd.atoms.positions.copy()
        E = xprm.positions.copy()

        # Covariance matrix (same as your original)
        bal = np.dot(S.T, E)
        i = np.ones((S.shape[0], 1))
        lab1 = np.dot(S.T, i)
        lab2 = np.dot(i.T, E)
        C = (bal - np.dot(lab1, lab2) / S.shape[0]) / (E.shape[0] - 1)

        # Construct symmetric 4×4 matrix
        M = np.array([
            [C[0, 0] + C[1, 1] + C[2, 2], C[1, 2] - C[2, 1], C[2, 0] - C[0, 2], C[0, 1] - C[1, 0]],
            [C[1, 2] - C[2, 1], C[0, 0] - C[1, 1] - C[2, 2], C[0, 1] + C[1, 0], C[2, 0] + C[0, 2]],
            [C[2, 0] - C[0, 2], C[0, 1] + C[1, 0], -C[0, 0] + C[1, 1] - C[2, 2], C[1, 2] + C[2, 1]],
            [C[0, 1] - C[1, 0], C[2, 0] + C[0, 2], C[1, 2] + C[2, 1], -C[0, 0] - C[1, 1] + C[2, 2]]
        ])

        # Quaternion from largest eigenvalue (symmetric -> eigh is stable)
        w, v = np.linalg.eigh(M)
        q = v[:, np.argmax(w)]
        # match sign convention used previously (optional)
        if q[0] < 0:
            q = -q

        # Rotation matrix from quaternion
        R = np.array([
            [q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3],
             2*(q[1]*q[2] - q[0]*q[3]),
             2*(q[1]*q[3] + q[0]*q[2])],
            [2*(q[2]*q[1] + q[0]*q[3]),
             q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3],
             2*(q[2]*q[3] - q[0]*q[1])],
            [2*(q[3]*q[1] - q[0]*q[2]),
             2*(q[3]*q[2] + q[0]*q[1]),
             q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3]]
        ])

        o = xprm_cog - np.dot(stnd_cog, R.T)
        return o, R

    # bpFrame, bpParam, localHeliParam, stepFrame copied / adapted from your original
    def bpFrame(a, b, base_ori, base_orient):
        oribp = (base_ori[a, :] + base_ori[b, :]) / 2
        matbp = (base_orient[a, :].reshape(3, 3) + base_orient[b, :].reshape(3, 3)) / 2
        return oribp, matbp.reshape(9)

    def bpParam(o1, R1, o2, R2):
        a1 = np.cross(R2[:, 1], R1[:, 1])
        gamma = a1 / np.linalg.norm(a1)

        BuckleOpeningAngle = np.degrees(angle(R2[:, 1], R1[:, 1]))

        R1p = np.dot(rotMatrix(gamma, np.radians(-BuckleOpeningAngle / 2)), R1)
        R2p = np.dot(rotMatrix(gamma, np.radians(BuckleOpeningAngle / 2)), R2)

        Rm = (R1p + R2p) / 2
        om = (o1 + o2) / 2

        propeller = np.degrees(angle(R2p[:, 0], R1p[:, 0]))
        if np.dot(np.cross(R2p[:, 0], R1p[:, 0]), Rm[:, 1]) < 0:
            propeller = -propeller

        phase = np.degrees(angle(gamma, Rm[:, 0]))
        if np.dot(np.cross(gamma, Rm[:, 0]), Rm[:, 1]) < 0:
            phase = -phase

        buckle = BuckleOpeningAngle * np.cos(np.radians(phase))
        opening = BuckleOpeningAngle * np.sin(np.radians(phase))

        shear, stretch, stagger = np.dot(o1 - o2, Rm)

        return om, Rm.reshape(9), shear, stretch, stagger, buckle, propeller, opening

    def localHeliParam(o1, R1, o2, R2):
        a1 = np.cross(R2[:, 0] - R1[:, 0], R2[:, 1] - R1[:, 1])
        localaxis = a1 / np.linalg.norm(a1)

        TipInclAngle = np.degrees(angle(localaxis, R1[:, 2]))
        hinge_axis = np.cross(localaxis, R1[:, 2])
        norm_hinge_axis = hinge_axis / np.linalg.norm(hinge_axis)
        H1 = np.dot(rotMatrix(norm_hinge_axis, np.radians(-TipInclAngle)), R1)

        TipInclAngle2 = np.degrees(angle(localaxis, R2[:, 2]))
        hinge_axis2 = np.cross(localaxis, R2[:, 2])
        norm_hinge_axis2 = hinge_axis2 / np.linalg.norm(hinge_axis2)
        H2 = np.dot(rotMatrix(norm_hinge_axis2, np.radians(-TipInclAngle)), R2)

        Hm = (H1 + H2) / 2

        htwist = np.degrees(angle(H1[:, 1], H2[:, 1]))
        hrise = np.dot(o2 - o1, localaxis)

        phase_angle = np.degrees(angle(norm_hinge_axis, H1[:, 1]))
        if np.dot(np.cross(norm_hinge_axis, Hm[:, 1]), localaxis) < 0:
            phase_angle = -phase_angle

        tip = TipInclAngle * np.cos(np.radians(phase_angle))
        incl = TipInclAngle * np.sin(np.radians(phase_angle))

        AB = (o2 - o1) - hrise * localaxis
        AB = np.array(AB).reshape(3,)
        angleAD = 90 - htwist * 0.5
        AD = np.dot(rotMatrix(localaxis, np.radians(angleAD)), AB.T)

        norm_AD = AD / np.linalg.norm(AD)
        norm_AD = norm_AD.T

        magnitude_AD = np.linalg.norm(AB) * 0.5 / np.sin(np.radians(0.5 * htwist))
        o1_h = o1 + magnitude_AD * norm_AD
        o2_h = o1_h + hrise * localaxis

        xdisp = np.dot(o1 - o1_h, H1[:, 0])
        ydisp = np.dot(o1 - o1_h, H1[:, 1])

        return xdisp, ydisp, hrise, incl, tip, htwist

    def stepFrame(o1, R1, o2, R2):
        a1 = np.cross(R1[:, 2], R2[:, 2])
        hinge_axis = a1 / np.linalg.norm(a1)

        RollTiltAngle = np.degrees(angle(R1[:, 2], R2[:, 2]))

        R1p = np.dot(rotMatrix(hinge_axis, np.radians(RollTiltAngle / 2)), R1)
        R2p = np.dot(rotMatrix(hinge_axis, np.radians(-RollTiltAngle / 2)), R2)

        Rm = (R1p + R2p) / 2
        om = (o1 + o2) / 2

        shift, slide, rise = np.dot(o2 - o1, Rm)

        twist = np.degrees(angle(R1[:, 1], R2[:, 1]))
        if np.dot(np.cross(R1p[:, 1], R2p[:, 1]), Rm[:, 2]) < 0:
            twist = -twist

        phase = np.degrees(angle(hinge_axis, Rm[:, 1]))
        if np.dot(np.cross(hinge_axis, Rm[:, 1]), Rm[:, 2]) < 0:
            phase = -phase

        roll = RollTiltAngle * np.cos(np.radians(phase))
        tilt = RollTiltAngle * np.sin(np.radians(phase))

        return om, Rm.reshape(9), shift, slide, rise, tilt, roll, twist

    # geometry sizes (assume residue count constant across frames)
    total_res = len(u.residues)
    basepairs = total_res // 2
    steps = basepairs - 1

    # number of frames
    try:
        n_frames = len(u.trajectory)
    except Exception:
        # If Universe has no trajectory object or len fails, fall back to 1
        n_frames = 1

    if verbose:
        print(f"Total residues: {total_res}, basepairs: {basepairs}, steps: {steps}, frames: {n_frames}")

    # allocate outputs
    basepair_parameters_all = np.zeros((n_frames, basepairs, 6))
    step_parameters_all = np.zeros((n_frames, steps, 6))
    local_parameters_all = np.zeros((n_frames, steps, 6))

    # Frame loop
    pbar = tqdm(enumerate(u.trajectory if trajectory is not None else [u.trajectory.ts]))
    for fi, ts in pbar:
        # Temporary arrays for this frame
        base_ori = np.zeros((total_res, 3))
        base_orient = np.zeros((total_res, 9))

        # compute base origins & orientations for all residues (except maybe last, original used range(total_res-1))
        for res in range(total_res - 1):
            resname = u.residues[res].resname
            o, R = lsfit(resname, res, u.residues)
            base_ori[res, :] = o
            base_orient[res, :] = R.reshape(9)

            # flip orientation for second half (same logic as your original)
            if res >= total_res // 2:
                base_mat = base_orient[res, :].reshape(3, 3)
                base_mat[:, 1] *= -1
                base_mat[:, 2] *= -1
                base_orient[res, :] = base_mat.reshape(9)

        # Base-pair params
        for i in range(basepairs):
            base_a = i
            base_b = total_res - i - 1
            R1 = base_orient[base_a, :].reshape(3, 3)
            R2 = base_orient[base_b, :].reshape(3, 3)
            o1 = base_ori[base_a, :]
            o2 = base_ori[base_b, :]

            om, Rm, she, st, sta, buc, pro, ope = bpParam(o1, R1, o2, R2)
            basepair_parameters_all[fi, i, :] = (she, st, sta, buc, pro, ope)

        # Step and local parameters
        for i in range(steps):
            o1 = (base_ori[i, :]+base_ori[total_res - i - 1, :]) / 2  # not used directly here
            o1_bp = (base_ori[i, :]+base_ori[total_res - i - 1, :]) / 2  # same as your bp origin
            # but your step calculations use bp_ori values; compute as in original:
            o1_bp = None  # placeholder, we'll use i and i+1 on computed bp arrays below

        # Build bp origin & orientation arrays as original did (to feed step/local)
        bp_ori = np.zeros((basepairs, 3))
        bp_orient = np.zeros((basepairs, 9))
        for i in range(basepairs):
            ev = (base_ori[i, :] + base_ori[total_res - i - 1, :]) / 2
            na = (base_orient[i, :].reshape(3, 3) + base_orient[total_res - i - 1, :].reshape(3, 3)) / 2
            bp_ori[i, :] = ev
            bp_orient[i, :] = na.reshape(9)

        # compute step/local using bp arrays
        for i in range(steps):
            o1 = bp_ori[i, :]
            R1 = bp_orient[i, :].reshape(3, 3)
            o2 = bp_ori[i + 1, :]
            R2 = bp_orient[i + 1, :].reshape(3, 3)

            xdsp, ydsp, hrs, incl, tip, htw = localHeliParam(o1, R1, o2, R2)
            # ensure plain python floats
            local_parameters_all[fi, i, :] = (float(np.asarray(xdsp).item() if hasattr(xdsp, 'item') else xdsp),
                                              float(np.asarray(ydsp).item() if hasattr(ydsp, 'item') else ydsp),
                                              float(hrs), float(incl), float(tip), float(htw))

            om, Rm, sh, sl, ri, ti, ro, tw = stepFrame(o1, R1, o2, R2)
            step_parameters_all[fi, i, :] = (sh, sl, ri, ti, ro, tw)

        if verbose and ((fi % 10 == 0) or fi == n_frames - 1):
            print(f"Processed frame {fi+1}/{n_frames}")

    # # optionally save
    # if outfile_prefix:
    #     np.savez_compressed(f"{outfile_prefix}_heliparams.npz",
    #                         basepair_parameters=basepair_parameters_all,
    #                         step_parameters=step_parameters_all,
    #                         local_parameters=local_parameters_all)
    #     if verbose:
    #         print(f"Saved results to {outfile_prefix}_heliparams.npz")

    return basepair_parameters_all, step_parameters_all, local_parameters_all
