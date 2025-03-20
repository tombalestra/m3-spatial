import numpy as np
import math

def euler_to_direction(pitch, yaw, distance=1.0):
    """
    Convert pitch and yaw angles to a direction vector.
    
    Args:
        pitch (float): Pitch angle in degrees (up/down)
        yaw (float): Yaw angle in degrees (left/right)
        distance (float): Length of the direction vector
        
    Returns:
        np.array: Direction vector (x, y, z)
    """
    # Convert angles to radians
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)
    
    # Calculate direction vector
    x = distance * math.sin(yaw_rad) * math.cos(pitch_rad)
    y = distance * math.sin(pitch_rad)
    z = distance * math.cos(yaw_rad) * math.cos(pitch_rad)
    
    return np.array([x, y, z])

def direction_to_euler(direction):
    """
    Convert a direction vector to pitch and yaw angles.
    
    Args:
        direction (np.array): Direction vector (x, y, z)
        
    Returns:
        tuple: (pitch, yaw) angles in degrees
    """
    # Normalize the direction vector
    direction = direction / np.linalg.norm(direction)
    x, y, z = direction
    
    # Calculate pitch (elevation) angle
    pitch = math.degrees(math.asin(y))
    
    # Calculate yaw (azimuth) angle
    yaw = math.degrees(math.atan2(x, z))
    
    return pitch, yaw

def create_rotation_matrix(pitch, yaw, roll):
    """
    Create a rotation matrix from pitch, yaw, and roll angles.
    Uses the YXZ convention: first rotate around Y (yaw), then X (pitch), then Z (roll).
    
    Args:
        pitch (float): Pitch angle in degrees (around X axis)
        yaw (float): Yaw angle in degrees (around Y axis)
        roll (float): Roll angle in degrees (around Z axis)
        
    Returns:
        np.array: 3x3 rotation matrix
    """
    # Convert angles to radians
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)
    roll_rad = math.radians(roll)
    
    # Sin and cos values
    sp = math.sin(pitch_rad)
    cp = math.cos(pitch_rad)
    sy = math.sin(yaw_rad)
    cy = math.cos(yaw_rad)
    sr = math.sin(roll_rad)
    cr = math.cos(roll_rad)
    
    # Create rotation matrix - YXZ order
    # This matches what most 3D applications use for Euler angles
    R = np.array([
        [cy*cr + sp*sy*sr, -cy*sr + sp*sy*cr, cp*sy],
        [cp*sr, cp*cr, -sp],
        [-sy*cr + sp*cy*sr, sy*sr + sp*cy*cr, cp*cy]
    ])
    
    return R

def extract_euler_angles(R):
    """
    Extract pitch, yaw, and roll angles from a rotation matrix.
    Assumes YXZ convention.
    
    Args:
        R (np.array): 3x3 rotation matrix
        
    Returns:
        tuple: (pitch, yaw, roll) angles in degrees
    """
    # Check for gimbal lock
    if abs(R[1, 2]) > 0.99999:
        # Gimbal lock - pitch is +/- 90 degrees
        pitch = -90 if R[1, 2] > 0 else 90
        
        # In gimbal lock, yaw and roll are linked
        # Convention: set roll to 0 and compute yaw
        roll = 0
        yaw = math.degrees(math.atan2(R[0, 1], R[2, 1]))
    else:
        # No gimbal lock
        pitch = math.degrees(math.asin(-R[1, 2]))
        yaw = math.degrees(math.atan2(R[0, 2], R[2, 2]))
        roll = math.degrees(math.atan2(R[1, 0], R[1, 1]))
    
    # Ensure angles are in the range [-180, 180]
    pitch = ((pitch + 180) % 360) - 180
    yaw = ((yaw + 180) % 360) - 180
    roll = ((roll + 180) % 360) - 180
    
    return pitch, yaw, roll

def posrot_to_pdb(pos_rot, options=None):
    """
    Convert Position-Rotation format to PDB camera format.
    
    Args:
        pos_rot (dict): Position and rotation in Euler angles with keys:
                        x, y, z, pitch, yaw, roll
        options (dict): Optional camera parameters
                        
    Returns:
        dict: PDB camera format with keys:
              FoVx, FoVy, R, T, trans, scale
    """
    if options is None:
        options = {}
    
    # Extract parameters with defaults
    x = pos_rot.get('x', 0)
    y = pos_rot.get('y', 0)
    z = pos_rot.get('z', 0)
    pitch = pos_rot.get('pitch', 0)
    yaw = pos_rot.get('yaw', 0)
    roll = pos_rot.get('roll', 0)
    
    fovX = options.get('fovX', 1.4)
    fovY = options.get('fovY', 0.87)
    scale = options.get('scale', 1.0)
    
    # Create world rotation matrix from Euler angles
    R_world = create_rotation_matrix(pitch, yaw, roll)
    
    # Convert to camera space where camera looks along -Z
    # In camera space, +Z is backward, +Y is up, +X is right
    R_cam_from_world = np.array([
        [1, 0, 0],
        [0, -1, 0],  # Flip Y
        [0, 0, -1]   # Flip Z
    ])
    
    # Final camera rotation matrix
    R = R_cam_from_world @ R_world
    
    # Create PDB camera object
    return {
        'FoVx': fovX,
        'FoVy': fovY,
        'R': R.tolist(),
        'T': [x, y, z],
        'trans': [0, 0, 0],
        'scale': scale
    }

def pdb_to_posrot(pdb_camera):
    """
    Convert PDB camera format to Position-Rotation format.
    
    Args:
        pdb_camera (dict): PDB camera parameters
                          
    Returns:
        dict: Position and rotation in Euler angles with keys:
              position (dict with x, y, z) and rotation (dict with pitch, yaw, roll)
    """
    # Extract parameters
    R = np.array(pdb_camera['R'])
    T = np.array(pdb_camera['T'])
    
    # Convert from camera space back to world space
    R_world_from_cam = np.array([
        [1, 0, 0],
        [0, -1, 0],  # Flip Y back
        [0, 0, -1]   # Flip Z back
    ])
    
    # World rotation matrix
    R_world = R_world_from_cam @ R
    
    # Extract Euler angles
    pitch, yaw, roll = extract_euler_angles(R_world)
    
    # Extract position from translation vector
    position = {
        'x': float(T[0]),
        'y': float(T[1]),
        'z': float(T[2])
    }
    
    # Round rotation angles to 2 decimal places
    rotation = {
        'pitch': round(pitch, 2),
        'yaw': round(yaw, 2),
        'roll': round(roll, 2)
    }
    
    return {
        'position': position,
        'rotation': rotation
    }

def posrot_flat_to_nested(flat_posrot):
    """
    Convert a flat posrot dict to a nested structure.
    
    Args:
        flat_posrot (dict): Flat posrot with keys x, y, z, pitch, yaw, roll
        
    Returns:
        dict: Nested posrot with position and rotation dicts
    """
    return {
        'position': {
            'x': flat_posrot['x'],
            'y': flat_posrot['y'],
            'z': flat_posrot['z']
        },
        'rotation': {
            'pitch': flat_posrot['pitch'],
            'yaw': flat_posrot['yaw'],
            'roll': flat_posrot['roll']
        }
    }

def posrot_nested_to_flat(nested_posrot):
    """
    Convert a nested posrot dict to a flat structure.
    
    Args:
        nested_posrot (dict): Nested posrot with position and rotation dicts
        
    Returns:
        dict: Flat posrot with keys x, y, z, pitch, yaw, roll
    """
    return {
        'x': nested_posrot['position']['x'],
        'y': nested_posrot['position']['y'],
        'z': nested_posrot['position']['z'],
        'pitch': nested_posrot['rotation']['pitch'],
        'yaw': nested_posrot['rotation']['yaw'],
        'roll': nested_posrot['rotation']['roll']
    }

def matrix_similarity(A, B, tolerance=1e-6):
    """
    Check if two matrices are similar within a tolerance.
    
    Args:
        A (np.array): First matrix
        B (np.array): Second matrix
        tolerance (float): Maximum allowed difference
        
    Returns:
        bool: True if matrices are similar
    """
    return np.allclose(A, B, atol=tolerance)

def compare_pdb_cameras(original, recovered, tolerance_position=0.01, tolerance_rotation=1e-6, tolerance_fov=0.01):
    """
    Compare two PDB camera objects for similarity.
    
    Args:
        original (dict): Original PDB camera
        recovered (dict): Recovered PDB camera
        tolerance_position (float): Maximum position difference
        tolerance_rotation (float): Maximum rotation matrix difference
        tolerance_fov (float): Maximum FoV difference
        
    Returns:
        tuple: (bool, dict) - Success flag and error details
    """
    errors = {}
    
    # Compare position (T)
    T_error = np.max(np.abs(np.array(original['T']) - np.array(recovered['T'])))
    errors['position_error'] = T_error
    
    # Compare rotation matrix (R)
    R_error = np.max(np.abs(np.array(original['R']) - np.array(recovered['R'])))
    errors['rotation_error'] = R_error
    
    # Compare FoV
    fovX_error = abs(original['FoVx'] - recovered['FoVx'])
    fovY_error = abs(original['FoVy'] - recovered['FoVy'])
    errors['fov_error'] = max(fovX_error, fovY_error)
    
    # Check if all errors are within tolerance
    success = (T_error <= tolerance_position and 
               R_error <= tolerance_rotation and
               max(fovX_error, fovY_error) <= tolerance_fov)
    
    return success, errors

def angle_diff(a, b):
    """
    Calculate the shortest angular distance between two angles.
    
    Args:
        a (float): First angle in degrees
        b (float): Second angle in degrees
        
    Returns:
        float: Shortest angular distance in degrees
    """
    return abs(((a - b + 180) % 360) - 180)

def test_posrot_to_pdb_to_posrot(test_cases, tolerance_position=0.01, tolerance_rotation=5.0):
    """
    Test posrot → PDB → posrot transformations.
    
    Args:
        test_cases (list): List of posrot dictionaries
        tolerance_position (float): Maximum position error
        tolerance_rotation (float): Maximum rotation error in degrees
        
    Returns:
        bool: True if all tests pass
    """
    print("\n===== Testing posrot → PDB → posrot =====")
    print(f"Position tolerance: {tolerance_position}")
    print(f"Rotation tolerance: {tolerance_rotation} degrees")
    
    all_passed = True
    
    for i, original_posrot in enumerate(test_cases):
        print(f"\nTest case {i+1}: {original_posrot}")
        
        # Convert to PDB format
        pdb_camera = posrot_to_pdb(original_posrot)
        
        # Convert back to posrot
        recovered_posrot = pdb_to_posrot(pdb_camera)
        recovered_flat = posrot_nested_to_flat(recovered_posrot)
        
        print(f"Recovered: {recovered_flat}")
        
        # Calculate position error
        position_error = max(
            abs(original_posrot['x'] - recovered_posrot['position']['x']),
            abs(original_posrot['y'] - recovered_posrot['position']['y']),
            abs(original_posrot['z'] - recovered_posrot['position']['z'])
        )
        
        # Calculate rotation error
        rotation_error = max(
            angle_diff(original_posrot['pitch'], recovered_posrot['rotation']['pitch']),
            angle_diff(original_posrot['yaw'], recovered_posrot['rotation']['yaw']),
            angle_diff(original_posrot['roll'], recovered_posrot['rotation']['roll'])
        )
        
        print(f"Position error: {position_error:.4f}")
        print(f"Rotation error: {rotation_error:.4f} degrees")
        
        test_passed = True
        
        if position_error > tolerance_position:
            print(f"❌ FAILED: Position error exceeds {tolerance_position}")
            test_passed = False
        
        if rotation_error > tolerance_rotation:
            print(f"❌ FAILED: Rotation error exceeds {tolerance_rotation} degrees")
            test_passed = False
            
        if test_passed:
            print("✓ PASSED: Error within tolerance")
        else:
            all_passed = False
    
    return all_passed

def test_pdb_to_posrot_to_pdb(pdb_test_cases, tolerance_position=0.01, tolerance_rotation=1e-6, tolerance_fov=0.01):
    """
    Test PDB → posrot → PDB transformations.
    
    Args:
        pdb_test_cases (list): List of PDB camera dictionaries
        tolerance_position (float): Maximum position error
        tolerance_rotation (float): Maximum rotation matrix error
        tolerance_fov (float): Maximum FoV error
        
    Returns:
        bool: True if all tests pass
    """
    print("\n===== Testing PDB → posrot → PDB =====")
    print(f"Position tolerance: {tolerance_position}")
    print(f"Rotation matrix tolerance: {tolerance_rotation}")
    print(f"FoV tolerance: {tolerance_fov}")
    
    all_passed = True
    
    for i, original_pdb in enumerate(pdb_test_cases):
        print(f"\nTest case {i+1}")
        
        # Convert to posrot format
        posrot = pdb_to_posrot(original_pdb)
        
        # Convert back to PDB format
        flat_posrot = posrot_nested_to_flat(posrot)
        recovered_pdb = posrot_to_pdb(flat_posrot, {
            'fovX': original_pdb['FoVx'],
            'fovY': original_pdb['FoVy'],
            'scale': original_pdb['scale']
        })
        
        # Compare original and recovered PDB cameras
        success, errors = compare_pdb_cameras(
            original_pdb, recovered_pdb, 
            tolerance_position, tolerance_rotation, tolerance_fov
        )
        
        print(f"Position error: {errors['position_error']:.8f}")
        print(f"Rotation matrix error: {errors['rotation_error']:.8f}")
        print(f"FoV error: {errors['fov_error']:.8f}")
        
        if success:
            print("✓ PASSED: Error within tolerance")
        else:
            print("❌ FAILED: Error exceeds tolerance")
            if errors['position_error'] > tolerance_position:
                print(f"  Position error exceeds {tolerance_position}")
            if errors['rotation_error'] > tolerance_rotation:
                print(f"  Rotation error exceeds {tolerance_rotation}")
            if errors['fov_error'] > tolerance_fov:
                print(f"  FoV error exceeds {tolerance_fov}")
            all_passed = False
    
    return all_passed

def generate_pdb_test_cases():
    test_cases = []

    # Basic test case (looking forward)
    test_cases.append({
        'FoVx': 1.4,
        'FoVy': 0.87,
        'R': create_rotation_matrix(0, 0, 0).tolist(),
        'T': [0, 0, 5],
        'trans': [0, 0, 0],
        'scale': 1.0
    })

    # Test case looking left (yaw = 90 degrees)
    test_cases.append({
        'FoVx': 1.4,
        'FoVy': 0.87,
        'R': create_rotation_matrix(0, 90, 0).tolist(),
        'T': [0, 0, 5],
        'trans': [0, 0, 0],
        'scale': 1.0
    })

    # Random cases from known Euler angles (guaranteed compatibility)
    np.random.seed(42)  # for reproducibility
    for _ in range(5):
        pitch = np.random.uniform(-90, 90)
        yaw = np.random.uniform(-180, 180)
        roll = np.random.uniform(-180, 180)
        R = create_rotation_matrix(pitch, yaw, roll)
        test_cases.append({
            'FoVx': np.random.uniform(1.2, 1.8),
            'FoVy': np.random.uniform(0.7, 1.1),
            'R': R.tolist(),
            'T': np.random.uniform(-5, 5, size=3).tolist(),
            'trans': [0, 0, 0],
            'scale': np.random.uniform(0.5, 2.0)
        })

    return test_cases

def test_transformations(tolerance_position=0.01, tolerance_rotation=5.0,
                        tolerance_matrix=1e-6, tolerance_fov=0.01):
    """
    Test bidirectional transformations with various camera parameters.
    """
    print("\n===== BIDIRECTIONAL TRANSFORMATION TESTING =====")
    
    # Test cases for posrot → PDB → posrot
    posrot_test_cases = [
        {'x': 0, 'y': 0, 'z': 5, 'pitch': 0, 'yaw': 0, 'roll': 0},      # Looking forward
        {'x': 0, 'y': 0, 'z': 5, 'pitch': 0, 'yaw': 90, 'roll': 0},     # Looking right
        {'x': 0, 'y': 0, 'z': 5, 'pitch': 45, 'yaw': 0, 'roll': 0},     # Looking up
        {'x': 0, 'y': 0, 'z': 5, 'pitch': 30, 'yaw': 60, 'roll': 15},   # Complex rotation
        {'x': 0, 'y': 0, 'z': 5, 'pitch': 89.9, 'yaw': 45, 'roll': 0},  # Near gimbal lock
        {'x': 0.83, 'y': 0.42, 'z': 4.72, 'pitch': 28.67, 'yaw': 178.18, 'roll': -3.10},
        {'x': 1.25, 'y': -0.87, 'z': 3.96, 'pitch': 15.43, 'yaw': -45.62, 'roll': 8.75},
        {'x': -2.14, 'y': 1.03, 'z': 6.28, 'pitch': -22.89, 'yaw': 90.33, 'roll': -12.47},
        {'x': 0.56, 'y': 2.18, 'z': 5.34, 'pitch': 45.12, 'yaw': 135.76, 'roll': 30.22},
        {'x': 3.42, 'y': -1.56, 'z': 7.83, 'pitch': -30.66, 'yaw': -105.82, 'roll': -25.18},
        {'x': -0.92, 'y': 0.77, 'z': 3.21, 'pitch': 60.35, 'yaw': -172.44, 'roll': 5.93},
        {'x': 2.63, 'y': 1.48, 'z': 8.15, 'pitch': -75.28, 'yaw': 25.91, 'roll': -18.37},
        {'x': -1.37, 'y': -2.25, 'z': 4.95, 'pitch': 10.84, 'yaw': -65.19, 'roll': 15.63},
        {'x': 0.51, 'y': 3.12, 'z': 6.73, 'pitch': 55.49, 'yaw': 87.62, 'roll': -7.29},
        {'x': 4.18, 'y': -0.68, 'z': 5.27, 'pitch': -43.15, 'yaw': 155.78, 'roll': 22.41},
        {'x': -3.25, 'y': 2.31, 'z': 9.46, 'pitch': 38.27, 'yaw': -38.63, 'roll': -15.92},
    ]
    
    # Test cases for PDB → posrot → PDB
    pdb_test_cases = generate_pdb_test_cases()
    
    # Run tests
    posrot_to_pdb_to_posrot_passed = test_posrot_to_pdb_to_posrot(
        posrot_test_cases, tolerance_position, tolerance_rotation
    )
    
    pdb_to_posrot_to_pdb_passed = test_pdb_to_posrot_to_pdb(
        pdb_test_cases, tolerance_position, tolerance_matrix, tolerance_fov
    )
    
    # Overall result
    all_passed = posrot_to_pdb_to_posrot_passed and pdb_to_posrot_to_pdb_passed
    
    print("\n===== OVERALL TEST RESULTS =====")
    print(f"posrot → PDB → posrot: {'✓ PASSED' if posrot_to_pdb_to_posrot_passed else '❌ FAILED'}")
    print(f"PDB → posrot → PDB: {'✓ PASSED' if pdb_to_posrot_to_pdb_passed else '❌ FAILED'}")
    print(f"Overall: {'✓ All tests passed' if all_passed else '❌ Some tests failed'}")
    
    return all_passed

if __name__ == "__main__":
    test_transformations(
        tolerance_position=0.01,    # position is fine
        tolerance_rotation=5.0,     # degrees (fine)
        tolerance_matrix=1e-4,      # rotation matrices (relaxed)
        tolerance_fov=0.01          # FoV is fine
    )
