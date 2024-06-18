# 표준 라이브러리 임포트
import os
import sys
import json
import zlib
import io
import struct
import re
import uuid
import subprocess
from datetime import datetime
from importlib.metadata import distribution, PackageNotFoundError
from concurrent.futures import ThreadPoolExecutor, as_completed

# 이미지 처리
from tqdm import tqdm
from PIL import Image, PngImagePlugin
import piexif
import piexif.helper

# 데이터베이스 및 ORM
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, ForeignKey, DateTime, Text, LargeBinary
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID, ENUM

# 웹 서비스 및 클라우드 스토리지
from google.cloud import storage

# 네트워크/데이터베이스 구성 및 접근
from sshtunnel import SSHTunnelForwarder
from urllib.parse import quote_plus

# 동시성 및 멀티스레딩
import threading

def install_requirements():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def get_installed_packages():
    installed_packages = set()
    try:
        for dist in distribution().discover():
            installed_packages.add(dist.metadata['Name'].lower())
    except PackageNotFoundError:
        pass
    return installed_packages

try:
    with open('requirements.txt', 'r') as f:
        packages = f.readlines()
    installed_packages = get_installed_packages()
    missing_packages = [pkg.split('==')[0].strip() for pkg in packages if pkg.split('==')[0].strip().lower() not in installed_packages]

    if missing_packages:
        print("Installing missing packages:", missing_packages)
        install_requirements()
except Exception as e:
    print(f"An error occurred while installing requirements: {e}")


Base = declarative_base()

engine = None
session_factory = None
server = None
lock = threading.Lock()

class PNGInfoAPI:
    def read_info_from_image(self, image: Image.Image):
        IGNORED_INFO_KEYS = {
            'jfif', 'jfif_version', 'jfif_unit', 'jfif_density', 'dpi', 'exif',
            'loop', 'background', 'timestamp', 'duration', 'progressive', 'progression',
            'icc_profile', 'chromaticity', 'photoshop',
        }
        
        items = (image.info or {}).copy()
        geninfo = items.pop('parameters', None)
    
        if "exif" in items:
            exif_data = items.pop("exif")
            try:
                exif = piexif.load(exif_data)
            except OSError:
                exif = None
            exif_comment = (exif or {}).get("Exif", {}).get(piexif.ExifIFD.UserComment, b'')
            try:
                exif_comment = piexif.helper.UserComment.load(exif_comment)
            except ValueError:
                exif_comment = exif_comment.decode('utf8', errors="ignore")

            if exif_comment:
                items['exif comment'] = exif_comment
                geninfo = exif_comment
        
        elif "comment" in items: # for gif
            geninfo = items["comment"].decode('utf8', errors="ignore")

        for field in IGNORED_INFO_KEYS:
            items.pop(field, None)


        if items.get("Software", None) == "NovelAI":
            try:
                json_info = json.loads(items["Comment"])
                geninfo = f"""{items["Description"]}
Negative prompt: {json_info["uc"]}
Steps: {json_info["steps"]}, CFG scale: {json_info["scale"]}, Seed: {json_info["seed"]}, Size: {image.width}x{image.height}, Clip skip: 2, ENSD: 31337"""
            except Exception as e:
                print('NovelAI 정보 처리 중 오류 발생:', e)
        
        return geninfo


    def parse_generation_parameters(self, x: str):
        res = {}
        lines = x.strip().split("\n")  # 입력된 문자열을 줄 단위로 분리

        for i, line in enumerate(lines):  # 각 줄과 그 인덱스에 대해 반복
            line = line.strip()  # 현재 줄의 앞뒤 공백 제거
            if i == 0:  # 첫 번째 줄인 경우
                res["Prompt"] = line
            elif i == 1 and line.startswith("Negative prompt:"):  # 두 번째 줄이며 "Negative prompt:"로 시작하는 경우
                res["Negative prompt"] = line[16:].strip()
            elif i == 2:  # 세 번째 줄인 경우, 옵션들을 처리
                # 여기에서 각 키-값에 대한 매칭 작업을 수행합니다.
                keys = [
                    "Steps", "Sampler", "CFG scale", "Seed", "Size", 
                    "Model hash", "Model", "VAE hash", "VAE", 
                    "Denoising strength", "Clip skip", "Hires upscale",
                    "Hires upscaler", 
                ]
                for key in keys:
                    # 정규 표현식을 사용하여 각 키에 해당하는 값을 찾습니다.
                    match = re.search(fr'{key}: ([^,]+),', line)
                    if match:
                        # 찾은 값은 그룹 1에 있습니다.
                        value = match.group(1).strip()
                        res[key] = value
                
                controlnet_patterns = re.findall(r'ControlNet \d+: "(.*?)"', line, re.DOTALL)
                for idx, cn_content in enumerate(controlnet_patterns):
                    # ControlNet 내부의 키-값 쌍을 추출합니다.
                    cn_dict = {}
                    cn_pairs = re.findall(r'(\w+): ([^,]+)', cn_content)
                    for key, value in cn_pairs:
                        cn_dict[key.strip()] = value.strip()
                    res[f"ControlNet {idx}"] = cn_dict

        return res

    def geninfo_params(self, image):
        try:
            geninfo = self.read_info_from_image(image)
            if geninfo == None:
                params = None
                
                return geninfo, params
            else:
                params = self.parse_generation_parameters(geninfo)
            return geninfo, params
        except Exception as e:
            print("Error:", str(e))

class Resource(Base):
    __tablename__ = 'resource'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=True)
    original_resource_id = Column(Integer, nullable=True)
    history_id = Column(Integer, nullable=True)
    category_id = Column(Integer, nullable=True)
    folder_id = Column(Integer, nullable=True)
    name = Column(String(1000), default="")
    description = Column(Text, default="")
    image = Column(String(200), default="")
    generation_data = Column(Text, default="")
    model_name = Column(String(200), default="")
    model_hash = Column(String(100), default="")
    sampler = Column(String(100), default="Euler")
    sampler_scheduler = Column(String(100), default="")
    prompt = Column(Text, default="")
    negative_prompt = Column(Text, default="")
    width = Column(Integer, default=512)
    height = Column(Integer, default=512)
    steps = Column(Integer, default=20)
    cfg_scale = Column(Float, default=7.5)
    seed = Column(Integer, default=-1)
    is_highres = Column(Boolean, default=False)
    hr_upscaler = Column(String(300), default="")
    hr_steps = Column(Integer, default=0)
    hr_denoising_strength = Column(Float, default=0)
    hr_upscale_by = Column(Float, default=1)
    is_display = Column(Boolean, default=True)
    is_empty = Column(Boolean, default=False)
    for_testing = Column(Boolean, default=False)
    sd_vae = Column(String(200), default="")
    is_bmab = Column(Boolean, default=False)
    is_i2i = Column(Boolean, default=False)
    resize_mode = Column(Integer, default=0)
    init_image = Column(String(200), default="")
    i2i_denoising_strength = Column(Float, default=0)
    is_sd_upscale = Column(Boolean, default=False)
    sd_tile_overlap = Column(Integer, default=0)
    sd_scale_factor = Column(Integer, default=0)
    sd_upscale = Column(String(4000), default="")
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4)
    thumbnail_image = Column(String(200), default="")
    thumbnail_image_512 = Column(String(300), default="")
    is_variation = Column(Boolean, default=False)
    generate_opt = Column(String(200), default="Upload")
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

class PublicFolder(Base):
    __tablename__ = 'public_folder'
    
    id = Column(Integer, primary_key=True)
    create_user_id = Column(Integer, nullable=True)
    json_file = Column(Text, nullable=True)  # Text type used to store file path
    team_id = Column(Integer, nullable=True)
    status = Column(ENUM('AV', 'UN', name='folderstatus'), default='AV', nullable=False)

class SdModel(Base):
    __tablename__ = 'sdmodel'

    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    model_name = Column(String(200), nullable=False)
    hash = Column(String(10), index=True, nullable=False)
    sha256 = Column(String(64), nullable=False)
    thumbnail_image = Column(String(200))  # 이미지 파일 경로를 저장
    is_active = Column(Boolean, default=False)
    folder_id = Column(Integer, nullable=True)

class FolderResourceMap(Base):
    __tablename__ = 'folder_resource_mapping'

    id = Column(Integer, primary_key=True)
    folder_json_id = Column(String(255), nullable=False)
    resource_id = Column(Integer, ForeignKey('resource.id'), nullable=True)
    nano_id = Column(String(21), nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

def start_ssh_tunnel():
    server = SSHTunnelForwarder(
        ('34.64.105.81', 22),
        ssh_username='nerdystar',
        ssh_private_key='./wcidfu-ssh',
        remote_bind_address=('10.1.31.44', 5432)
    )
    server.start()
    print("SSH tunnel established")
    return server

def setup_database_engine(password, port):
    db_user = "wcidfu"
    db_host = "127.0.0.1"
    db_name = "wcidfu"
    encoded_password = quote_plus(password)
    engine = create_engine(f'postgresql+psycopg2://{db_user}:{encoded_password}@{db_host}:{port}/{db_name}')
    Base.metadata.create_all(engine)
    return engine

def get_session():
    global engine, session_factory, server, lock
    with lock:
        if server is None:
            server = start_ssh_tunnel()
        if engine is None:
            engine = setup_database_engine("nerdy@2024", server.local_bind_port)
        if session_factory is None:
            session_factory = sessionmaker(bind=engine)
    session = scoped_session(session_factory)  # 수정된 부분: 세션 팩토리에서 직접 scoped_session 인스턴스 생성
    return session, server

def end_session(session):
    session.close()

def upload_to_bucket(blob_name, data, bucket_name):
    current_script_path = os.path.abspath(__file__)
    base_directory = os.path.dirname(current_script_path)
    
    credentials_path = os.path.join(base_directory, 'wcidfu-77f802b00777.json')
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(data)
    clean_blob_name = blob_name.replace("_media/", "")    
    return clean_blob_name

def print_timestamp(message):
    print(f"{message}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def create_resource(user_id, original_image, image_128, image_518, session):
    pnginfo = PngImagePlugin.PngInfo()
    png_info_api_instance = PNGInfoAPI()
    geninfo, params = png_info_api_instance.geninfo_params(image=original_image)

    generation_data = geninfo
    pnginfo.add_text("parameters", json.dumps(geninfo))

    try:
        new_resource = Resource(user_id=user_id)
        session.add(new_resource)
        session.commit()

        # JSON 청크 데이터 생성
        chunk_data_json = json.dumps({
            "my_uuid": str(new_resource.uuid),
            "id": str(new_resource.id),
            "created_at": str(new_resource.created_at),
            "controlnet_uuid": ""
        })
        chunk_data = chunk_data_json.encode('utf-8')
        chunk_type = b'nsSt'
        crc = zlib.crc32(chunk_type + chunk_data)
        custom_chunk = struct.pack('>I', len(chunk_data)) + chunk_type + chunk_data + struct.pack('>I', crc)
        
        # 원본 이미지에 사용자 정의 청크 삽입
        original_image_buffer = io.BytesIO()
        original_image.save(original_image_buffer, format="PNG", pnginfo= pnginfo)
        original_image_data = original_image_buffer.getvalue()
        iend_index = original_image_data.rfind(b'IEND')
        modified_img_data = original_image_data[:iend_index-4] + custom_chunk + original_image_data[iend_index-4:]

        # 수정된 이미지 데이터를 버킷에 업로드
        modified_blob_name = f"_media/resource/{str(new_resource.uuid)}.png"
        original_image_url = upload_to_bucket(modified_blob_name, modified_img_data, "wcidfu-bucket")
        
        new_resource.image = original_image_url  # 이미지 URL을 리소스 객체에 저장
        session.commit()
        
        # 썸네일 이미지 처리
        image_128_buffer = io.BytesIO()
        image_128.save(image_128_buffer, format="PNG")
        image_128_blob_name = f"_media/resource_thumbnail/{str(new_resource.uuid)}_128.png"
        thumbnail_128_image_url = upload_to_bucket(image_128_blob_name, image_128_buffer.getvalue(), "wcidfu-bucket")
        
        new_resource.thumbnail_image = thumbnail_128_image_url  # 썸네일 URL을 리소스 객체에 저장

        # 썸네일 이미지 처리
        image_518_buffer = io.BytesIO()
        image_518.save(image_518_buffer, format="PNG")
        image_518_blob_name = f"_media/thumbnail_512/{str(new_resource.uuid)}_512.png"
        thumbnail_518_image_url = upload_to_bucket(image_518_blob_name, image_518_buffer.getvalue(), "wcidfu-bucket")
        
        new_resource.thumbnail_image_512 = thumbnail_518_image_url  # 썸네일 URL을 리소스 객체에 저장

        session.commit()
        
        resource_uuid = str(new_resource.uuid)
        resource_id = new_resource.id
        if geninfo == None and params == None:
            return resource_id

        else:    
            if generation_data:
                new_resource.generation_data = generation_data
                session.commit()
            else:
                pass
            
            if "Prompt" in params:
                new_resource.prompt = params["Prompt"]
                session.commit()
            else:
                pass

            if "Negative prompt" in params:
                new_resource.negative_prompt = params["Negative prompt"]
                session.commit()
            else:
                pass

            if "Steps" in params:
                new_resource.steps = params["Steps"]
                session.commit()
            else:
                pass

            if "Sampler" in params:
                new_resource.sampler = params["Sampler"]
                session.commit()
            else:
                pass

            if "CFG scale" in params:
                new_resource.cfg_scale = params["CFG scale"]
                session.commit()
            else:
                pass

            if "Seed" in params:
                new_resource.seed = params["Seed"]
                session.commit()
            else:
                pass

            if "Size" in params:
                size_list = [int(n) for n in params["Size"].split('x')]
                new_resource.height = size_list[0]
                new_resource.width = size_list[1]
                session.commit()
            else:
                pass

            if "Model hash" in params:
                sd_model = session.query(SdModel).filter_by(hash=params["Model hash"]).first()
                if sd_model:
                    new_resource.model_hash = sd_model.hash
                    new_resource.model_name = sd_model.model_name
                else:
                    new_resource.model_hash = params["Model hash"]
                    new_resource.model_name = params["Model"]
            else:
                pass

            if "VAE" in params:
                new_resource.sd_vae = params["VAE"]
                session.commit()
            else:
                pass

            if "Denoising strength" in params:
                new_resource.is_highres = True
                new_resource.hr_denoising_strength = params["Denoising strength"]
                session.commit()

                if "Hires upscale" in params:
                    new_resource.hr_upscale_by = params["Hires upscale"]
                    session.commit()
                if "Hires upscaler" in params:
                    new_resource.hr_upscaler = params["Hires upscaler"]
                    session.commit()
            else:
                pass
        session.commit()
        return resource_id
    except Exception as e:
        session.rollback()  # 에러 발생 시 롤백
        raise e
    finally:
        session.close()
    return

def process_images(image_path, heights=(128, 512)):
    def resize_image(original_image, height):
        aspect_ratio = original_image.width / original_image.height
        new_width = int(height * aspect_ratio)
        resized_image = original_image.resize((new_width, height))
        image_bytes = io.BytesIO()
        resized_image.save(image_bytes, format='PNG')
        image_bytes.seek(0)
        return Image.open(image_bytes)

    original_image = Image.open(image_path)
    resized_images = {height: resize_image(original_image, height) for height in heights}
    return original_image, resized_images[128], resized_images[512]

def mapping_folder_resource(session, resource_id, folder_json_id, team_id):
    new_mapping = FolderResourceMap(
            folder_json_id = folder_json_id,
            resource_id =resource_id,
            nano_id = team_id
        )
    session.add(new_mapping)
    session.commit()
    
def main(upload_folder, user_id, folder_json_id, team_id):
    png_files = [f for f in os.listdir(upload_folder) if f.endswith('.png')]
    print_timestamp('[main.py 작동 시작]')
    session, server = get_session()
    try:
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_and_upload, png, upload_folder, user_id, folder_json_id, team_id, session): png for png in png_files}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
                png = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing {png}: {e}")
    
    finally:
        end_session(session)
        print_timestamp('[main.py 작동 종료]')

def process_and_upload(png, upload_folder, user_id, folder_json_id, team_id, session):
    png_path = os.path.join(upload_folder, png)
    original_image, image_128, image_518 = process_images(png_path)  # 이미지 크기 조정 및 저장
    resource_id = create_resource(user_id, original_image, image_128, image_518, session)
    mapping_folder_resource(session, resource_id, folder_json_id, team_id)

if __name__ == "__main__":
    # 명령줄 인수 파싱
    if len(sys.argv) != 5:
        print("사용법: folder_uploader.py <upload_folder> <user_id> <folder_json_id> <team_id>")
        sys.exit(1)
    
    upload_folder, user_id, folder_json_id, team_id = sys.argv[1:]
    main(upload_folder, user_id, folder_json_id, team_id)

# AXjJkzww4t1uSfuxr2-Mr
# '/Users/nerdystar/Desktop/Sejuani'
# 8bBUcHEJ0noEPDpyak9yp