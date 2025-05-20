#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hydrogen Bond and Tilt Angle Analysis for Functionalized OH Groups on SiO2 Surface

Modifications in this version:
  1. Global cutoff variables are defined so that the hydrogen bonding cutoff (HB_CUTOFF)
     and the O1–H cutoff (OH_CUTOFF) can be adjusted in one place.
  2. For each MD frame, the hydrogen bond (O2 acceptor) is recalculated for each OH group
     so that the number of valid hydrogen bonds is updated every frame.
  3. Progress reporting is added to show how many frames have been analyzed.
  
This script:
  - Reads XDATCAR (with non-orthogonal cells and periodic boundaries).
  - Adjusts z coordinates (mapping z in [0.9,1) to negative values, shifting upward, and recentering so that the slab center is 0.5).
  - In the first frame, identifies OH groups (each H bonded to an oxygen within OH_CUTOFF).
  - For every frame:
       • Calculates the O1–H bond length and tilt angle (for every OH group).
       • Recalculates the hydrogen bond acceptor (O2) for each OH group and, if found,
         validates the hydrogen bond if H–O2 distance < HB_CUTOFF and the O1–H–O2 angle is between 120 deg and 180 deg.
  - Writes per-frame and overall averages.
"""

import numpy as np

# Global cutoff parameters (in Ang)
OH_CUTOFF = 1.2  # O1–H bond cutoff
HB_CUTOFF = 3.0  # Hydrogen bond cutoff (used both in searching and validating)

def read_all_frames(filename="XDATCAR"):
    """
    Reads all frames from XDATCAR.
    
    Returns:
      - frames: list of numpy arrays (each shape: (total_atoms, 3)) in fractional coordinates.
      - lattice: 3x3 numpy array of Cartesian lattice vectors.
      - atom_types: list of element symbols for each atom.
    """
    with open(filename, "r") as f:
        lines = f.readlines()
    
    scaling_factor = float(lines[1].strip())
    lattice = np.array([np.array(list(map(float, lines[i].split()))) * scaling_factor for i in range(2, 5)])
    elements_line = lines[5].split()
    counts = list(map(int, lines[6].split()))
    total_atoms = sum(counts)
    
    # Build a list of atom types based on header.
    atom_types = []
    for elem, count in zip(elements_line, counts):
        atom_types.extend([elem] * count)
    
    frames = []
    i = 8
    while i < len(lines):
        if "configuration" in lines[i].lower():
            i += 1
        if i + total_atoms > len(lines):
            break
        frame_positions = []
        for j in range(total_atoms):
            pos_line = lines[i+j].split()
            pos = np.array(list(map(float, pos_line[:3])))
            frame_positions.append(pos)
        frames.append(np.array(frame_positions))
        i += total_atoms
    return frames, lattice, atom_types

def apply_pbc(diff, lattice):
    """
    Applies the minimum image convention on a difference vector in fractional coordinates,
    and converts it to Cartesian coordinates.
    """
    diff = diff - np.round(diff)
    return np.dot(diff, lattice)

def find_closest_neighbor_of_type(target_index, positions, lattice, atom_types, desired_type, exclude=None):
    """
    Finds the closest neighbor of the given desired_type for the atom at target_index.
    """
    if exclude is None:
        exclude = []
    target = positions[target_index]
    min_dist = float('inf')
    closest_index = None
    for i, pos in enumerate(positions):
        if i == target_index or i in exclude:
            continue
        if atom_types[i].upper() != desired_type.upper():
            continue
        diff = pos - target
        cart_diff = apply_pbc(diff, lattice)
        dist = np.linalg.norm(cart_diff)
        if dist < min_dist:
            min_dist = dist
            closest_index = i
    return closest_index, min_dist

def find_hbond_acceptor(H_index, positions, lattice, atom_types, O1_index, cutoff=HB_CUTOFF):
    """
    For a given H atom, finds a hydrogen bond acceptor oxygen (O2) within a cutoff.
    Returns the index of the closest acceptable oxygen and its distance.
    """
    target = positions[H_index]
    min_dist = float('inf')
    acceptor_index = None
    for i, pos in enumerate(positions):
        if i == O1_index:
            continue
        if atom_types[i].upper() != "O":
            continue
        diff = pos - target
        cart_diff = apply_pbc(diff, lattice)
        dist = np.linalg.norm(cart_diff)
        if dist < cutoff and dist < min_dist:
            min_dist = dist
            acceptor_index = i
    return acceptor_index, min_dist

def compute_angle(vec1, vec2):
    """
    Computes the angle (in degrees) between two vectors.
    """
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return None
    cos_angle = dot / (norm1 * norm2)
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)

def adjust_z_coordinates(positions):
    """
    Adjusts the z coordinates of atomic positions to account for periodicity along z:
      1. For atoms with z in [0.9, 1), subtract 1.
      2. Shift the entire slab upward by adding 0.5.
      3. Compute the geometric center and recenter so that the center is at 0.5.
    
    Parameters:
        positions (np.array): Array of shape (N, 3) with fractional coordinates.
    
    Returns:
        np.array: New array with adjusted z coordinates.
    """
    adjusted_positions = positions.copy()
    z = adjusted_positions[:, 2]
    high_z_mask = (z >= 0.9) & (z < 1.0)
    adjusted_positions[high_z_mask, 2] = z[high_z_mask] - 1.0
    adjusted_positions[:, 2] += 0.5
    z_center = np.mean(adjusted_positions[:, 2])
    adjusted_positions[:, 2] += (0.5 - z_center)
    return adjusted_positions

def compute_tilt_angle(bond_vec, orientation):
    """
    Computes the tilt angle of the O1–H bond with respect to the (x,y) plane.
    
    Parameters:
        bond_vec (np.array): O1–H bond vector in Cartesian coordinates.
        orientation (int): +1 for "upper", -1 for "bottom".
    
    Returns:
        float: Tilt angle in degrees.
        
    Procedure:
      1. Compute raw magnitude: angle_magnitude = degrees(arctan2(|dz|, sqrt(dx²+dy²))).
      2. Raw sign = +1 if dz >= 0, else -1.
      3. For "upper" groups: final_angle = angle_magnitude * raw_sign.
         For "bottom" groups: final_angle = angle_magnitude * raw_sign * (-1).
    """
    xy_norm = np.linalg.norm(bond_vec[:2])
    if xy_norm < 1e-8:
        return 0.0
    angle_magnitude = np.degrees(np.arctan2(abs(bond_vec[2]), xy_norm))
    raw_sign = 1 if bond_vec[2] >= 0 else -1
    if orientation == 1:
        final_angle = angle_magnitude * raw_sign
    else:
        final_angle = angle_magnitude * raw_sign * (-1)
    return final_angle

def identify_OH_groups(first_frame, lattice, atom_types, OH_cutoff=OH_CUTOFF, hbond_cutoff=HB_CUTOFF):
    """
    In the first frame, identifies OH groups.
    
    For each H atom:
      - Finds the nearest oxygen (O1) within OH_cutoff.
      - Stores the indices even if a hydrogen bond acceptor (O2) is not found.
      - Determines the group orientation based on adjusted H_z:
          If H_z > 0.5, orientation = +1 ("upper"); otherwise -1 ("bottom").
    
    Returns:
        List of tuples: (H_index, O1_index, orientation)
    """
    groups = []
    for i, elem in enumerate(atom_types):
        if elem.upper() == "H":
            O1_index, dist_OH = find_closest_neighbor_of_type(i, first_frame, lattice, atom_types, "O", exclude=[i])
            if O1_index is not None and dist_OH < OH_cutoff:
                orientation = 1 if first_frame[i][2] > 0.5 else -1
                groups.append((i, O1_index, orientation))
    return groups

def analyze_frames(frames, lattice, atom_types, OH_groups, total_frames):
    """
    For each frame, compute:
      - For every OH group (all groups), the O1–H bond length and tilt angle.
      - For each OH group, recalc the hydrogen bond acceptor (O2) and, if found,
        compute the H–O₂ distance and O1–H–O2 angle.
        Only if H–O2 distance < HB_CUTOFF and angle is between 120 deg and 180 deg,
        the hydrogen bond is considered valid.
    
    Also prints progress updates.
    
    Returns:
      - frame_results: List of per-frame dictionaries.
      - overall: Dictionary with overall averages.
      
    Each per-frame dictionary includes:
        'frame': frame number,
        'avg_OH': average O1–H bond length over all OH groups,
        'avg_tilt': average tilt angle over all OH groups,
        'hb_count': count of valid hydrogen bonds,
        'avg_HO2': average H–O2 distance for valid hydrogen bonds,
        'avg_angle': average O1–H–O2 angle for valid hydrogen bonds,
        'OH_count': total number of OH groups.
    """
    frame_results = []
    all_OH_lengths = []
    all_tilt_angles = []
    valid_HO2_distances = []
    valid_angles = []
    total_valid_hb = 0

    for frame_index, positions in enumerate(frames):
        OH_lengths = []
        tilt_angles = []
        frame_HO2 = []
        frame_angles = []
        valid_count = 0
        
        # For each OH group (using indices from the first frame)
        for (H_idx, O1_idx, orientation) in OH_groups:
            # Compute O1–H bond length.
            diff_OH = positions[H_idx] - positions[O1_idx]
            d_OH = np.linalg.norm(apply_pbc(diff_OH, lattice))
            OH_lengths.append(d_OH)
            
            # Compute tilt angle.
            bond_vec = apply_pbc(positions[H_idx] - positions[O1_idx], lattice)
            tilt_angle = compute_tilt_angle(bond_vec, orientation)
            tilt_angles.append(tilt_angle)
            
            # Recalculate hydrogen bond acceptor for this frame.
            O2_index, _ = find_hbond_acceptor(H_idx, positions, lattice, atom_types, O1_idx, cutoff=HB_CUTOFF)
            if O2_index is not None:
                diff_HO2 = positions[O2_index] - positions[H_idx]
                d_HO2 = np.linalg.norm(apply_pbc(diff_HO2, lattice))
                # Compute O1–H–O₂ angle.
                vec_HO1 = apply_pbc(positions[O1_idx] - positions[H_idx], lattice)
                vec_HO2 = apply_pbc(positions[O2_index] - positions[H_idx], lattice)
                angle_OH_O2 = compute_angle(vec_HO1, vec_HO2)
                
                # Validate hydrogen bond criteria.
                if d_HO2 < HB_CUTOFF and angle_OH_O2 is not None and (120 <= angle_OH_O2 <= 180):
                    frame_HO2.append(d_HO2)
                    frame_angles.append(angle_OH_O2)
                    valid_count += 1

        avg_OH = np.mean(OH_lengths) if OH_lengths else None
        avg_tilt = np.mean(tilt_angles) if tilt_angles else None
        avg_HO2 = np.mean(frame_HO2) if frame_HO2 else None
        avg_angle = np.mean(frame_angles) if frame_angles else None
        
        frame_results.append({
            'frame': frame_index + 1,
            'avg_OH': avg_OH,
            'avg_tilt': avg_tilt,
            'hb_count': valid_count,
            'avg_HO2': avg_HO2,
            'avg_angle': avg_angle,
            'OH_count': len(OH_lengths)
        })
        
        all_OH_lengths.extend(OH_lengths)
        all_tilt_angles.extend(tilt_angles)
        valid_HO2_distances.extend(frame_HO2)
        valid_angles.extend(frame_angles)
        total_valid_hb += valid_count

        # Progress reporting.
        if (frame_index + 1) % 1000 == 0 or (frame_index + 1) == total_frames:
            percent = 100 * (frame_index + 1) / total_frames
            print(f"Processed frame {frame_index + 1}/{total_frames} ({percent:.1f}%).")
    
    overall = {
        'avg_OH': np.mean(all_OH_lengths) if all_OH_lengths else None,
        'avg_tilt': np.mean(all_tilt_angles) if all_tilt_angles else None,
        'total_valid_hb': total_valid_hb,
        'avg_HO2': np.mean(valid_HO2_distances) if valid_HO2_distances else None,
        'avg_angle': np.mean(valid_angles) if valid_angles else None,
        'total_OH': len(all_OH_lengths)
    }
    return frame_results, overall

def write_frame_averages(frame_results, filename="frame_averages.dat"):
    """
    Writes per-frame averages:
      - Average O1–H bond length and tilt angle (over all OH groups).
      - Count of valid hydrogen bonds, average H–O₂ distance, and average O1–H–O₂ angle (for valid hydrogen bonds).
    """
    with open(filename, "w") as f:
        f.write("#Frame\tAvg_O1-H\tAvg_Tilt\tOH_count\tValid_HB_Count\tAvg_H-O2\tAvg_O1-H-O2_Angle\n")
        for res in frame_results:
            f.write(f"{res['frame']}\t"
                    f"{res['avg_OH'] if res['avg_OH'] is not None else 'NaN'}\t"
                    f"{res['avg_tilt'] if res['avg_tilt'] is not None else 'NaN'}\t"
                    f"{res['OH_count']}\t"
                    f"{res['hb_count']}\t"
                    f"{res['avg_HO2'] if res['avg_HO2'] is not None else 'NaN'}\t"
                    f"{res['avg_angle'] if res['avg_angle'] is not None else 'NaN'}\n")
    print(f"Frame averages written to {filename}")

def write_overall_average(overall, filename="overall_average.dat"):
    """
    Writes overall averages across all frames.
    """
    with open(filename, "w") as f:
        f.write("#Overall Average O1-H bond length, Tilt angle, Valid HB count, Avg H-O2, and Avg O1-H-O2 angle\n")
        f.write(f"Avg_O1-H: {overall['avg_OH'] if overall['avg_OH'] is not None else 'NaN'}\n")
        f.write(f"Avg_Tilt: {overall['avg_tilt'] if overall['avg_tilt'] is not None else 'NaN'}\n")
        f.write(f"Total Valid HBonds: {overall['total_valid_hb']}\n")
        f.write(f"Avg_H-O2: {overall['avg_HO2'] if overall['avg_HO2'] is not None else 'NaN'}\n")
        f.write(f"Avg_O1-H-O2_Angle: {overall['avg_angle'] if overall['avg_angle'] is not None else 'NaN'}\n")
        f.write(f"Total OH groups analyzed: {overall['total_OH']}\n")
    print(f"Overall averages written to {filename}")

def main():
    try:
        frames, lattice, atom_types = read_all_frames("XDATCAR")
    except Exception as e:
        print("Error reading XDATCAR:", e)
        return
    if not frames:
        print("No frames found in XDATCAR.")
        return

    total_frames = len(frames)
    # Adjust z coordinates for periodicity in every frame.
    frames = [adjust_z_coordinates(frame) for frame in frames]
    
    # Identify OH groups (O1–H pairs) in the first frame.
    first_frame = frames[0]
    OH_groups = identify_OH_groups(first_frame, lattice, atom_types, OH_cutoff=OH_CUTOFF, hbond_cutoff=HB_CUTOFF)
    if not OH_groups:
        print("No valid OH groups found in the first frame.")
        return
    print(f"Identified {len(OH_groups)} OH groups in the first frame.")
    
    # Analyze each frame using the OH groups from the first frame.
    frame_results, overall = analyze_frames(frames, lattice, atom_types, OH_groups, total_frames)
    write_frame_averages(frame_results, filename="frame_averages.dat")
    write_overall_average(overall, filename="overall_average.dat")

if __name__ == "__main__":
    main()
