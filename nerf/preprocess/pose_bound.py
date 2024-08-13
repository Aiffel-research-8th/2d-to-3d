import numpy as np  # 수치 계산을 위한 NumPy 라이브러리 임포트
import struct  # 바이너리 데이터 처리를 위한 struct 모듈 임포트
import collections  # namedtuple 사용을 위한 collections 모듈 임포트
import os  # 파일 경로 처리를 위한 os 모듈 임포트

# COLMAP에서 사용하는 카메라 모델 ID와 이름 매핑
CAMERA_MODELS = {
    0: 'SIMPLE_PINHOLE',
    1: 'PINHOLE',
    2: 'SIMPLE_RADIAL',
    3: 'RADIAL',
    4: 'OPENCV',
    5: 'OPENCV_FISHEYE',
    6: 'FULL_OPENCV',
    7: 'FOV',
    8: 'SIMPLE_RADIAL_FISHEYE',
    9: 'RADIAL_FISHEYE',
    10: 'THIN_PRISM_FISHEYE'
}

# 카메라 정보를 저장하기 위한 namedtuple 정의
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])

# 이미지 정보를 저장하기 위한 namedtuple 정의
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """
    바이너리 파일에서 지정된 바이트 수만큼 읽어 언팩하는 함수
    fid: 파일 객체
    num_bytes: 읽을 바이트 수
    format_char_sequence: struct 모듈의 포맷 문자열
    endian_character: 엔디안 지정 ("<" 는 리틀 엔디안)
    """
    data = fid.read(num_bytes)  # 지정된 바이트 수만큼 읽기
    return struct.unpack(endian_character + format_char_sequence, data)  # 읽은 데이터를 언팩

def read_cameras_binary(path_to_model_file):
    """COLMAP의 cameras.bin 파일을 읽어 카메라 정보를 딕셔너리로 반환하는 함수"""
    cameras = {}
    with open(path_to_model_file, "rb") as fid:  # 바이너리 모드로 파일 열기
        num_cameras = read_next_bytes(fid, 8, "Q")[0]  # 카메라 수 읽기
        for _ in range(num_cameras):  # 각 카메라에 대해 반복
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]  # 카메라 ID
            model_id = camera_properties[1]  # 카메라 모델 ID
            width = camera_properties[2]  # 이미지 너비
            height = camera_properties[3]  # 이미지 높이
            
            # 카메라 모델에 따른 파라미터 수 결정
            num_params = {
                'SIMPLE_PINHOLE': 3,
                'PINHOLE': 4,
                'SIMPLE_RADIAL': 4,
                'RADIAL': 5,
                'OPENCV': 8,
                'OPENCV_FISHEYE': 8,
                'FULL_OPENCV': 12,
                'FOV': 5,
                'SIMPLE_RADIAL_FISHEYE': 4,
                'RADIAL_FISHEYE': 5,
                'THIN_PRISM_FISHEYE': 12
            }[CAMERA_MODELS[model_id]]
            
            # 카메라 파라미터 읽기
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            
            # Camera namedtuple 생성 및 딕셔너리에 추가
            cameras[camera_id] = Camera(id=camera_id,
                                        model=CAMERA_MODELS[model_id],
                                        width=width,
                                        height=height,
                                        params=params)
    return cameras

def read_images_binary(path_to_model_file):
    """COLMAP의 images.bin 파일을 읽어 이미지 정보를 딕셔너리로 반환하는 함수"""
    images = {}
    with open(path_to_model_file, "rb") as fid:  # 바이너리 모드로 파일 열기
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]  # 이미지 수 읽기
        for _ in range(num_reg_images):  # 각 이미지에 대해 반복
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]  # 이미지 ID
            qvec = np.array(binary_image_properties[1:5])  # 쿼터니언 벡터
            tvec = np.array(binary_image_properties[5:8])  # 평행이동 벡터
            camera_id = binary_image_properties[8]  # 카메라 ID
            
            # 이미지 이름 읽기
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # null 문자를 만날 때까지 읽기
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            
            # 2D 점 정보 읽기
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(x_y_id_s[0::3]),
                                   tuple(x_y_id_s[1::3])])  # 2D 점 좌표
            point3D_ids = np.array(tuple(x_y_id_s[2::3]))  # 3D 점 ID
            
            # BaseImage namedtuple 생성 및 딕셔너리에 추가
            images[image_id] = BaseImage(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images

def qvec2rotmat(qvec):
    """쿼터니언을 회전 행렬로 변환하는 함수"""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def create_poses_bounds(cameras, images):
    """카메라와 이미지 정보를 이용해 포즈와 경계값을 생성하는 함수"""
    poses = []
    bounds = []
    for _, img in images.items():  # 각 이미지에 대해 반복
        R = qvec2rotmat(img.qvec)  # 쿼터니언을 회전 행렬로 변환
        t = img.tvec  # 평행이동 벡터
        c2w = np.eye(4)  # 4x4 단위 행렬 생성
        c2w[:3, :3] = R.T  # 회전 행렬의 전치를 할당
        c2w[:3, 3] = -R.T @ t  # 카메라 위치 계산
        
        camera = cameras[img.camera_id]  # 해당 이미지의 카메라 정보 가져오기
        
        # 카메라 모델에 따라 내부 파라미터 추출
        if camera.model == 'SIMPLE_PINHOLE':
            f = camera.params[0]
            cx, cy = camera.params[1], camera.params[2]
        elif camera.model == 'PINHOLE':
            fx, fy = camera.params[0], camera.params[1]
            cx, cy = camera.params[2], camera.params[3]
            f = (fx + fy) / 2
        elif camera.model == 'SIMPLE_RADIAL':
            f = camera.params[0]
            cx, cy = camera.params[1], camera.params[2]
        else:
            raise ValueError(f"Unsupported camera model: {camera.model}")

        h, w = camera.height, camera.width  # 이미지 높이와 너비
        
        # LLFF 형식에 맞게 포즈 매트릭스 구성
        pose = np.eye(4)  # 4x4 단위 행렬 생성
        pose[:3, :4] = c2w[:3, :4]  # 카메라에서 월드로의 변환 행렬 할당
        
        # 내부 파라미터를 포즈 매트릭스에 추가
        pose[3, 0] = f
        pose[3, 1] = cx
        pose[3, 2] = cy
        
        poses.append(pose)  # 포즈 리스트에 추가

        # near와 far 평면 계산
        near = 0.1 * f
        far = 1000 * f
        bounds.append([near, far])  # 경계값 리스트에 추가

    poses = np.array(poses).astype(np.float32)  # 포즈 배열로 변환
    bounds = np.array(bounds).astype(np.float32)  # 경계값 배열로 변환
    return poses, bounds

def save_poses_bounds(poses, bounds, output_path):
    """포즈와 경계값을 numpy 파일로 저장하는 함수"""
    assert(poses.shape[0] == bounds.shape[0])  # 포즈와 경계값의 수가 같은지 확인
    poses_bounds = np.concatenate([poses[:, :3, :4].reshape(-1, 12), 
                                   poses[:, 3, :3], 
                                   bounds], axis=1)  # 포즈와 경계값을 하나의 배열로 결합
    np.save(output_path, poses_bounds)  # numpy 파일로 저장

if __name__ == "__main__":
    base_dir = "../data/sparse/0"  # COLMAP 출력 디렉토리
    output_dir = "../data"  # 결과 저장 디렉토리

    # COLMAP 출력 파일 읽기
    cameras = read_cameras_binary(os.path.join(base_dir, "cameras.bin"))
    images = read_images_binary(os.path.join(base_dir, "images.bin"))

    # 포즈와 경계값 생성
    poses, bounds = create_poses_bounds(cameras, images)
    
    # 결과 저장
    save_poses_bounds(poses, bounds, os.path.join(output_dir, "poses_bounds3.npy"))

    # 결과 정보 출력
    print(f"poses shape: {poses.shape}")
    print(f"bounds shape: {bounds.shape}")
    print(f"poses_bounds.npy 파일이 {output_dir} 에 생성되었습니다.")