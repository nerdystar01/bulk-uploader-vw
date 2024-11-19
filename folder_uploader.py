# 표준 라이브러리 임포트
import os
import sys
import json
import zlib
import io
import struct
import time
import re
import uuid
import requests
from nanoid import generate
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
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, ForeignKey, DateTime, Text, Table
from sqlalchemy.orm import scoped_session, sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID, ENUM

# 웹 서비스 및 클라우드 스토리지
from google.cloud import storage
from google.oauth2 import service_account

# 네트워크/데이터베이스 구성 및 접근
from sshtunnel import SSHTunnelForwarder
from urllib.parse import quote_plus

# 동시성 및 멀티스레딩
import threading
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

MAX_RETRIES = 5
RETRY_DELAY = 5

class SSHConnection:
    def __init__(self):
        self.server = None

    def start_ssh_tunnel(self):
        try:
            self.server = SSHTunnelForwarder(
                ('34.64.105.81', 22),
                ssh_username='nerdystar',
                ssh_pkey='./wcidfu-ssh',
                remote_bind_address=('10.1.31.44', 5432),
                set_keepalive=60
            )
            self.server.start()
            logging.info("SSH tunnel established")
            return self.server
        except Exception as e:
            logging.error(f"Error establishing SSH tunnel: {str(e)}")
            raise

    def check_connection(self):
        if self.server is None or not self.server.is_active:
            logging.info("SSH connection is not active. Reconnecting...")
            self.start_ssh_tunnel()

    def stop_ssh_tunnel(self):
        if self.server:
            self.server.stop()
            logging.info("SSH tunnel closed")

ssh_connection = SSHConnection()

def retry_on_exception(func):
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                ssh_connection.check_connection()
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f"Error in {func.__name__}: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    logging.info(f"Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    raise
    return wrapper

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

resource_likes = Table('resource_likes', Base.metadata,
    Column('resource_id', Integer, ForeignKey('resource.id')),
    Column('user_id', Integer, ForeignKey('user.id'))
)
resource_tags = Table('resource_tags', Base.metadata,
    Column('resource_id', Integer, ForeignKey('resource.id')),
    Column('colorcodetagss_id', Integer, ForeignKey('color_code_tags.id'))
)
resource_hidden_users = Table(
    'resource_hidden_users',
    Base.metadata,
    Column('resource_id', Integer, ForeignKey('resource.id')),
    Column('user_id', Integer, ForeignKey('user.id')),
)

resource_tabbed_users = Table(
    'resource_tabbed_users',
    Base.metadata,
    Column('resource_id', Integer, ForeignKey('resource.id')),
    Column('user_id', Integer, ForeignKey('user.id')),
)
class User(Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    folder_id = Column(Integer, nullable=True)
    json_file = Column(String, nullable=True)  # SQLAlchemy does not support a direct FileField equivalent
    nano_id = Column(String(21), unique=True, nullable=True)
    liked_resources = relationship("Resource", secondary=resource_likes, back_populates="likes")
    hidden_resources = relationship(
        "Resource",
        secondary="resource_hidden_users",
        back_populates="hidden_by"
    )
    
    tabbed_resources = relationship(
        "Resource",
        secondary="resource_tabbed_users",
        back_populates="tabbed_by"
    )

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
    star_rating = Column(Integer, default=0)
    clip_skip = Column(Integer, default=0)
    tags = relationship("ColorCodeTags", secondary=resource_tags, back_populates="resources")
    generate_opt = Column(String(200), default="Upload")
    count_download = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    likes = relationship("User", secondary=resource_likes, back_populates="liked_resources")
    reference_resource_id = Column(Integer, nullable=True)
    royalty = Column(Float, default=0.0)
    hidden_by = relationship(
        "User",
        secondary="resource_hidden_users",
        back_populates="hidden_resources"
    )
    
    tabbed_by = relationship(
        "User",
        secondary="resource_tabbed_users",
        back_populates="tabbed_resources"
    )
    
    # GPT Vision 점수
    gpt_vision_score = Column(Integer, nullable=True)


class ColorCodeTags(Base):
    __tablename__ = 'color_code_tags'
    
    id = Column(Integer, primary_key=True)
    color_code = Column(String(7))
    tag = Column(String(4000))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Resource와의 역참조 관계
    resources = relationship("Resource", secondary=resource_tags, back_populates="tags")

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

class Team(Base):
    __tablename__ = 'team'

    id = Column(Integer, primary_key=True)
    create_user_id = Column(String(255), nullable=False)
    nano_id = Column(String(21), nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

@retry_on_exception
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

@retry_on_exception
def setup_database_engine(password, port):
    db_user = "wcidfu"
    db_host = "127.0.0.1"
    db_name = "wcidfu"
    encoded_password = quote_plus(password)
    engine = create_engine(f'postgresql+psycopg2://{db_user}:{encoded_password}@{db_host}:{port}/{db_name}')
    Base.metadata.create_all(engine)
    return engine

@retry_on_exception
def get_session():
    server = ssh_connection.start_ssh_tunnel()
    engine = setup_database_engine("nerdy@2024", server.local_bind_port)
    session_factory = sessionmaker(bind=engine)
    session = scoped_session(session_factory)
    return session, server

def end_session(session):
    session.close()
    ssh_connection.stop_ssh_tunnel()

@retry_on_exception
def upload_to_bucket(blob_name, data, bucket_name):
    current_script_path = os.path.abspath(__file__)
    base_directory = os.path.dirname(current_script_path)
    
    credentials_path = os.path.join(base_directory, 'wcidfu-77f802b00777.json')
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

    credentials = service_account.Credentials.from_service_account_file(credentials_path)

    storage_client = storage.Client(credentials = credentials)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(data)
    clean_blob_name = blob_name.replace("_media/", "")    
    return clean_blob_name

def print_timestamp(message):
    print(f"{message}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

@retry_on_exception
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
                schedule_label_list = ["Uniform", "Karras", "Exponential", "Polyexponential"]
                sampler = params["Sampler"].strip()
                scheduler_found = False
                
                for label in schedule_label_list:
                    if label.lower() in sampler.lower():
                        new_resource.sampler = sampler.replace(label, "").strip()
                        new_resource.sampler_scheduler = label
                        scheduler_found = True
                        break
                
                if not scheduler_found:
                    new_resource.sampler = sampler
                    new_resource.sampler_scheduler = None
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
                new_resource.width = size_list[0]
                new_resource.height = size_list[1]
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

            if "Clip skip" in params:
                new_resource.clip_skip = params["Clip skip"]
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

folder_resource_mapping = {}

def update_folder_resource_mapping(folder_id, resource_id):
    if folder_id not in folder_resource_mapping:
        folder_resource_mapping[folder_id] = []
    folder_resource_mapping[folder_id].append(resource_id)

@retry_on_exception
def mapping_folder_resource(session, resource_id, folder_json_id, team_id):
    new_mapping = FolderResourceMap(
            folder_json_id = folder_json_id,
            resource_id =resource_id,
            nano_id = team_id
        )
    session.add(new_mapping)
    session.commit()

@retry_on_exception
def process_and_upload(png, upload_folder, user_id, folder_json_id, team_id, session):
    png_path = os.path.join(upload_folder, png)
    original_image, image_128, image_518 = process_images(png_path)  # 이미지 크기 조정 및 저장
    resource_id = create_resource(user_id, original_image, image_128, image_518, session)
    mapping_folder_resource(session, resource_id, folder_json_id, team_id)
    update_folder_resource_mapping(folder_json_id, resource_id)

def process_folder(upload_folder, user_id, team_id, session):
    for root, dirs, files in os.walk(upload_folder):
        folder_json_id = os.path.basename(root)
        png_files = [f for f in files if f.endswith('.png')]
        if not png_files:
            continue
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(process_and_upload, png, root, user_id, folder_json_id, team_id, session): png for png in png_files}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing images in {folder_json_id}"):
                png = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing {png} in folder {folder_json_id}: {e}")
             
def generate_folder_id():
    return generate(size=21)

def create_folder_tree(path, parent_id=None, parent_id_list=[]):
    folder_name = os.path.abspath(path)
    folder_id = generate_folder_id()

    folder_info = {
        "id": folder_id,
        "name": folder_name,
        "children": [],
        "parentId": parent_id,
        "parentIdList": parent_id_list
    }

    parent_id_list_updated = parent_id_list + [folder_id] if parent_id else [folder_id]

    folder_dict = {folder_id: folder_info}
    
    # 하위 폴더만 찾기
    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    with tqdm(total=len(subdirs), desc=f"Processing folder structure {folder_name}") as pbar:
        for subdir in subdirs:
            subdir_path = os.path.join(path, subdir)
            child_folder_result = create_folder_tree(subdir_path, folder_id, parent_id_list_updated)
            child_folder_info = child_folder_result["folder_info"]
            folder_info["children"].append(child_folder_info["id"])
            folder_dict.update(child_folder_result["folder_dict"])
            pbar.update(1)

    return {
        "folder_info": folder_info,
        "folder_dict": folder_dict
    }

@retry_on_exception
def update_public_json_file(session, nano_id, uploaded_folder_structure):
    user = session.query(User).filter(User.nano_id == nano_id).first()
    team = session.query(Team).filter(Team.nano_id == nano_id).first()
    
    if user:
        file_path = user.json_file
        if not user.json_file:
            raise KeyError("User Private folder not found or no json file associated.")
        entity = user
        new_file_name = f"user_json_file/{generate(size=21)}.json"
    elif team:
        team_id = team.id
        public_folder = session.query(PublicFolder).filter(PublicFolder.team_id == team_id).first()
    
        if not public_folder or not public_folder.json_file:
            raise KeyError("Public folder not found or no json file associated.")
        file_path = public_folder.json_file
        entity = public_folder
        new_file_name = f"public_json_file/{generate(size=21)}.json"
    else:
        raise KeyError("Neither user nor team found with the given nano_id.")
    
    google_url = f"https://storage.googleapis.com/wcidfu-bucket/_media/{file_path}"

    # 기존 JSON 파일 가져오기
    response = requests.get(google_url)
    response.raise_for_status()  # 요청 실패 시 예외 발생

    # JSON 파일 내용 디코딩 및 로드
    folder_tree_file = response.content
    tree_file = json.loads(folder_tree_file.decode('utf-8'))

    # 현재 트리 구조에 새로운 폴더 구조 업데이트
    current_tree = tree_file.get('json', {})
    current_tree.update(uploaded_folder_structure)

    # Google Cloud Storage 클라이언트 생성
    current_script_path = os.path.abspath(__file__)
    base_directory = os.path.dirname(current_script_path)
    credentials_path = os.path.join(base_directory, 'wcidfu-77f802b00777.json')

    if os.path.exists(credentials_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        storage_client = storage.Client(credentials=credentials)
    else:
        raise FileNotFoundError(f"Credentials file not found: {credentials_path}")
    
    bucket_name = "wcidfu-bucket"
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"_media/{file_path}")

    # 기존 파일 백업
    backup_blob = bucket.blob(f"_media/backup/{file_path}")
    bucket.copy_blob(blob, bucket, new_name=backup_blob.name) 

    # 기존 파일 삭제
    blob.delete()

    # 새로운 JSON 파일 경로 설정 및 업로드
    new_blob = bucket.blob(f"_media/{new_file_name}")
    
    # 최종적으로 업데이트된 JSON 데이터를 생성하여 업로드
    updated_json_string = json.dumps(tree_file, ensure_ascii=False, indent=2)
    new_blob.upload_from_string(updated_json_string, content_type='application/json')

    # 데이터베이스 업데이트 및 세션 커밋
    entity.json_file = new_file_name
    session.commit()

    # 모든 작업이 성공하면 백업 파일 삭제
    backup_blob.delete()

def generate_folder_structure(root_path):
    result = create_folder_tree(root_path)
    folder_dict = result["folder_dict"]
    
    # 현재 날짜와 시간으로 파일 이름 생성
    now = datetime.now()
    file_name = now.strftime("%Y%m%d_%H%M%S") + "_folder_structure.json"
    
    # folder_tree 디렉토리 생성 (없는 경우)
    folder_tree_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "folder_tree")
    os.makedirs(folder_tree_path, exist_ok=True)
    
    # JSON 파일로 저장
    file_path = os.path.join(folder_tree_path, file_name)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(folder_dict, f, ensure_ascii=False, indent=2)
    
    print(f"폴더 구조가 {file_path}에 저장되었습니다.")
    
    return folder_dict

import json
import uuid
import requests
from google.cloud import storage

def process_folder_with_structure(folder_structure, root_path, user_id, nano_id, session):
    total_folders = len(folder_structure)
    folder_progress = tqdm(folder_structure.items(), total=total_folders, desc="Total folder processing")

    for folder_id, folder_info in folder_progress:
        folder_progress.set_description(f"Processing {folder_info['name']} (Remaining: {total_folders})")
        total_folders -= 1

        
        folder_path = folder_info['name']

        if not os.path.exists(folder_path):
            folder_progress.write(f"Skipping {folder_info['name']}: folder does not exist.")
            continue
        
        # 폴더 정보에서 이름만 추출
        folder_name = os.path.basename(folder_path)

        uploade_folder_structure = {}
        updated_folder_info = folder_info.copy()
        updated_folder_info['name'] = folder_name
        uploade_folder_structure[folder_id] = updated_folder_info
        
        png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        
        try:
            update_public_json_file(session, nano_id, uploade_folder_structure)
        except Exception as e:
            logging.error(f"Error updating public JSON file: {str(e)}")
            continue

        if not png_files:
            folder_progress.write(f"No PNG files in {folder_info['name']}, updating JSON.")
            continue

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(process_and_upload, png, folder_path, user_id, folder_id, nano_id, session): png for png in png_files}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Uploading images in {folder_info['name']}"):
                png = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing {png} in folder {folder_info['name']}: {e}")

        folder_progress.write(f"Completed processing {folder_info['name']}.")
    
    save_folder_resource_mapping(nano_id)

def save_folder_resource_mapping(nano_id):
    mapping_folder = "folder_resource_mapping"
    os.makedirs(mapping_folder, exist_ok=True)
    mapping_file = os.path.join(mapping_folder, f"{nano_id}_mapping.json")
    
    with open(mapping_file, 'w') as f:
        json.dump(folder_resource_mapping, f, indent=2)
    
    print(f"Folder resource mapping saved to {mapping_file}")

def main(upload_folder, user_id, nano_id):
    print_timestamp('[main.py 작동 시작]')
    session, server = get_session()
    
    try:
        folder_structure = generate_folder_structure(upload_folder)
        process_folder_with_structure(folder_structure, upload_folder, user_id, nano_id, session)
        
    finally:
        end_session(session)
        print_timestamp('[main.py 작동 종료]')

if __name__ == "__main__":
    # 명령줄 인수 파싱
    if len(sys.argv) != 4:
        print("사용법: folder_uploader.py <upload_folder> <user_id> <nano_id>")
        sys.exit(1)
    
    upload_folder, user_id, nano_id = sys.argv[1:]
    main(upload_folder, user_id, nano_id)

# AXjJkzww4t1uSfuxr2-Mr
# '/Users/nerdystar/Desktop/Sejuani'
# 8bBUcHEJ0noEPDpyak9yp