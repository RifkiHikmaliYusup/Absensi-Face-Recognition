from kivymd.app import MDApp
from kivy.core.window import Window
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, Rectangle, Ellipse
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivymd.uix.datatables import MDDataTable
from kivy.metrics import dp, sp
from kivymd.uix.label import MDLabel
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.textfield import MDTextField
from kivy.uix.screenmanager import ScreenManager, Screen
from kivymd.uix.button import MDRaisedButton
from datetime import datetime
from kivymd.uix.boxlayout import MDBoxLayout
from kivy_garden.mapview import MapView, MapMarker
from kivymd.uix.dialog import MDDialog
from kivy.metrics import dp, sp
from kivy.config import Config
from kivymd.uix.button import MDFlatButton
from kivy.animation import Animation
from kivy.uix.vkeyboard import VKeyboard
from kivy.uix.relativelayout import RelativeLayout
from kivymd.uix.selectioncontrol import MDCheckbox
from kivy.uix.scrollview import ScrollView
from kivymd.uix.card import MDCard
from kivy.core.text import LabelBase
from kivy.resources import resource_add_path
from kivy.uix.label import Label

import locale
import cv2
import requests
import math
import os
import json
import cvzone
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import time
import threading

# ==================== KONFIGURASI API ====================
# Ganti dengan IP address komputer backend jika akses dari device lain
API_BASE_URL = "http://localhost:5000"  # Untuk local
# API_BASE_URL = "http://192.168.1.10:5000"  # Untuk akses dari HP/tablet

API_ENDPOINTS = {
    'face_recognition': f"{API_BASE_URL}/api/v2/face-recognition",
    'register': f"{API_BASE_URL}/api/v2/face-recognition/register",
    'absen': f"{API_BASE_URL}/api/v2/absen",
    'karyawan': f"{API_BASE_URL}/api/v2/karyawan",
    'karyawan_delete': f"{API_BASE_URL}/api/v2/karyawan",
    'absensi_today': f"{API_BASE_URL}/api/v2/absensi/today",
    'all_karyawan': f"{API_BASE_URL}/api/v2/all-karyawan",
    'health': f"{API_BASE_URL}/health"
}

# ==================== KONFIGURASI APLIKASI ====================
# Folder untuk menyimpan gambar capture sementara
CAPTURE_FOLDER = "./captures"
os.makedirs(CAPTURE_FOLDER, exist_ok=True)

# Folder cache untuk peta
CACHE_FOLDER = "./cache"
os.makedirs(CACHE_FOLDER, exist_ok=True)

# Window size
Window.fullscreen = True
Window.size = (1080, 1920)

# ==================== LOAD MODEL ====================
model = YOLO("model/l_versions_3_100.pt")
classNames = ["fake", "real"]

app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1)

# ==================== VARIABEL GLOBAL ====================
confidence = 0.75
last_seen = {}
face_data = {}
loading_done = False
recognized_faces = {}
manual_lat = -6.866641  # Default latitude (Cimahi)
manual_lon = 107.5347632  # Default longitude (Cimahi)

# ==================== FUNGSI API ====================
def check_api_health():
    """Cek koneksi ke backend API"""
    try:
        response = requests.get(API_ENDPOINTS['health'], timeout=3)
        return response.status_code == 200
    except:
        return False

def load_face_data_from_api():
    """Load semua data wajah dari API"""
    global face_data, loading_done
    face_data.clear()
    
    try:
        response = requests.get(API_ENDPOINTS['face_recognition'], timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 200:
                for entry in data.get('data', []):
                    if 'nrp' in entry and 'face_encoding' in entry:
                        try:
                            # Parse JSON string ke list
                            encodings = json.loads(entry['face_encoding'])
                            face_data[entry['nrp']] = {
                                'encodings': np.array(encodings),
                                'name': entry.get('name', entry['nrp'])
                            }
                        except Exception as e:
                            print(f"⚠ Gagal parse face_encoding untuk {entry['nrp']}: {e}")
            
            print(f"✅ {len(face_data)} data wajah dimuat dari API")
        else:
            print(f"⚠ Gagal mengambil data dari API: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Tidak dapat terhubung ke API server")
    except Exception as e:
        print(f"❌ Error load data: {e}")
    
    loading_done = True

def send_absensi_to_api(nrp, image_path, latitude, longitude):
    """Mengirim data absensi ke API"""
    try:
        with open(image_path, "rb") as img_file:
            files = {"foto": (os.path.basename(image_path), img_file, "image/jpeg")}
            data = {
                "nrp": str(nrp),
                "latitude": str(latitude),
                "longitude": str(longitude)
            }
            
            response = requests.post(
                API_ENDPOINTS['absen'],
                data=data,
                files=files,
                timeout=15
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ API Error: {response.status_code} - {response.text}")
                return {
                    'status': response.status_code,
                    'message': f'Error {response.status_code}',
                    'data': None
                }
                
    except Exception as e:
        print(f"❌ Error send absensi: {e}")
        return {
            'status': 500,
            'message': str(e),
            'data': None
        }

def send_registration_to_api(nrp, encodings):
    """Mengirim data registrasi wajah ke API"""
    try:
        data = [{
            "nrp": str(nrp),
            "encodings": encodings
        }]
        
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(
            API_ENDPOINTS['register'],
            json=data,
            headers=headers,
            timeout=15
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error send registration: {e}")
        return None

def delete_karyawan_from_api(nrp_list):
    """Menghapus data karyawan dari API"""
    try:
        data = {"nrp_list": nrp_list}
        headers = {"Content-Type": "application/json"}
        
        response = requests.delete(
            API_ENDPOINTS['karyawan_delete'],
            json=data,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
            
    except Exception as e:
        print(f"❌ Error delete karyawan: {e}")
        return None

def get_absensi_today_from_api():
    """Mengambil data absensi hari ini dari API"""
    try:
        response = requests.get(API_ENDPOINTS['absensi_today'], timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('data', [])
        else:
            return []
            
    except Exception as e:
        print(f"❌ Error get absensi: {e}")
        return []

def get_karyawan_from_api(nrp=None):
    """Mengambil data karyawan dari API"""
    try:
        if nrp:
            url = f"{API_ENDPOINTS['karyawan']}/{nrp}"
        else:
            url = API_ENDPOINTS['all_karyawan']
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('data', []) if not nrp else data.get('data')
        else:
            return [] if not nrp else None
            
    except Exception as e:
        print(f"❌ Error get karyawan: {e}")
        return [] if not nrp else None

# ==================== FUNGSI LAINNYA ====================
def is_internet_available():
    try:
        requests.get("https://www.google.com", timeout=3)
        return True
    except:
        return False

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize_face(img, main_content):
    """Fungsi untuk mengenali wajah"""
    global recognized_faces, face_data, last_seen
    
    # Balik frame secara horizontal
    img = cv2.flip(img, 1)
    faces = app.get(img)

    for face in faces:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox

        frame_width = img.shape[1]
        mirrored_x1 = frame_width - x1
        face_encoding = np.array(face.normed_embedding)

        matched_nrps = []

        # Bandingkan wajah dengan data dari API
        for nrp, face_info in face_data.items():
            stored_encoding = face_info['encodings']
            similarity = cosine_similarity(face_encoding, stored_encoding)
            if similarity > 0.7:
                matched_nrps.append(nrp)

        # Kalau tidak ada wajah yang cocok → Unknown
        if not matched_nrps:
            recognized_faces["Unknown"] = (mirrored_x1, y1, time.time())
            print(f"❓ Wajah tidak dikenal")
        else:
            # Kalau ada wajah cocok → proses NRP
            for matched_nrp in matched_nrps:
                current_time = time.time()
                name = face_data.get(matched_nrp, {}).get('name', matched_nrp)

                # Cek apakah ini pertama kali wajah terlihat
                if matched_nrp not in last_seen:
                    last_seen[matched_nrp] = current_time
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_filename = f"{matched_nrp}_{timestamp}.jpg"
                    image_path_local = os.path.join(CAPTURE_FOLDER, image_filename)
                    cv2.imwrite(image_path_local, img)

                    print(f"📸 Kirim absensi - {name} ({matched_nrp})")
                    main_content.send_absensi(matched_nrp, image_path_local)

                else:
                    # Kalau sudah pernah terlihat, kirim lagi hanya setelah 5 detik
                    if current_time - last_seen[matched_nrp] >= 5:
                        last_seen[matched_nrp] = current_time
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_filename = f"{matched_nrp}_{timestamp}.jpg"
                        image_path_local = os.path.join(CAPTURE_FOLDER, image_filename)
                        cv2.imwrite(image_path_local, img)

                        print(f"📸 Kirim ulang absensi - {name} ({matched_nrp})")
                        main_content.send_absensi(matched_nrp, image_path_local)

                # Simpan posisi wajah yang dikenali
                recognized_faces[matched_nrp] = (mirrored_x1, y1, time.time())

# ==================== FONT ====================
font_dir = os.path.join(os.path.dirname(__file__), "assets", "font")
resource_add_path(font_dir)

try:
    LabelBase.register(name="Poppins-Bold", fn_regular=os.path.join(font_dir, "Poppins-Bold.ttf"))
    LabelBase.register(name="Poppins-Regular", fn_regular=os.path.join(font_dir, "Poppins-Regular.ttf"))
    LabelBase.register(name="Poppins-Medium", fn_regular=os.path.join(font_dir, "Poppins-Medium.ttf"))
    print("✅ Font Poppins berhasil dimuat")
except Exception as e:
    print(f"⚠ Font Poppins tidak ditemukan: {e}, menggunakan default")

# ==================== CAMERA SINGLETON ====================
class CameraSingleton:
    _instance = None
    _active_screen = None

    @staticmethod
    def get_instance(screen_name=None):
        if CameraSingleton._instance is None or not CameraSingleton._instance.isOpened():
            CameraSingleton._instance = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if CameraSingleton._instance.isOpened():
                CameraSingleton._active_screen = screen_name
                print(f"📷 Kamera diaktifkan untuk {screen_name}")
        return CameraSingleton._instance

    @staticmethod
    def release(screen_name=None):
        if CameraSingleton._instance is not None:
            if CameraSingleton._active_screen == screen_name:
                CameraSingleton._instance.release()
                CameraSingleton._instance = None
                CameraSingleton._active_screen = None
                print(f"📷 Kamera dilepaskan dari {screen_name}")

# ==================== SIDEBAR ====================
class Sidebar(BoxLayout):
    def __init__(self, screen_manager, **kwargs):
        super().__init__(**kwargs)
        self.screen_manager = screen_manager
        self.orientation = "vertical"
        self.size_hint = (0.2, 1)
        self.padding = 10
        self.pos_hint = {"top": 1}
        self.spacing = dp(20)
        self.active_button = None

        with self.canvas.before:
            Color(0.18, 0.30, 0.49, 1)
            self.rect = Rectangle(size=self.size, pos=self.pos)

        self.bind(size=self.update_rect, pos=self.update_rect)

        # Brand Logo
        self.brand_logo = Image(
            source="assets/logoHIGAIS2.png",
            size_hint=(None, None),
            size=(dp(120), dp(140)),
            keep_ratio=True,
            allow_stretch=True,
            pos_hint={"center_x": 0.5}
        )
        self.add_widget(self.brand_logo)

        self.add_widget(BoxLayout(size_hint_y=None, height=dp(30)))

        # Menu 1: Camera (Absensi)
        self.profile_logo = Image(
            source="assets/icon_camera.png",
            size_hint=(None, None),
            size=(dp(80), dp(80)),
            keep_ratio=True,
            allow_stretch=True,
            pos_hint={"center_x": 0.5}
        )
        self.profile_logo.bind(on_touch_down=self.on_camera_logo_pressed)
        self.add_widget(self.profile_logo)

        # Menu 2: Register
        self.register_logo = Image(
            source="assets/profile_icon.png",
            size_hint=(None, None),
            size=(dp(80), dp(80)),
            keep_ratio=True,
            allow_stretch=True,
            pos_hint={"center_x": 0.5}
        )
        self.register_logo.bind(on_touch_down=self.on_register_logo_pressed)
        self.add_widget(self.register_logo)

        # Menu 3: Remove
        self.remove_logo = Image(
            source="assets/remove_file.png",
            size_hint=(None, None),
            size=(dp(80), dp(80)),
            keep_ratio=True,
            allow_stretch=True,
            pos_hint={"center_x": 0.5}
        )
        self.remove_logo.bind(on_touch_down=self.delete_files)
        self.add_widget(self.remove_logo)

        # FR Text
        self.fr_text = Image(
            source="assets/FR_TEXT.png",
            size_hint=(None, None),
            size=(dp(100), dp(60)),
            allow_stretch=True,
            keep_ratio=True,
            pos_hint={"center_x": 0.5}
        )
        self.add_widget(BoxLayout())
        self.add_widget(self.fr_text)

    def update_rect(self, *args):
        self.rect.size = self.size
        self.rect.pos = self.pos

    def animate_button_click(self, button_instance):
        self.reset_button_states()
        anim = Animation(size=(dp(85), dp(85)), duration=0.1) + Animation(size=(dp(80), dp(80)), duration=0.1)
        anim.start(button_instance)
        
        with button_instance.canvas.after:
            Color(1, 1, 1, 0.4)
            self.circle = Ellipse(pos=button_instance.pos, size=button_instance.size)
        
        self.active_button = button_instance

    def reset_button_states(self):
        if self.active_button:
            self.active_button.canvas.after.clear()
            self.active_button = None

    def on_camera_logo_pressed(self, instance, touch):
        if not instance.collide_point(*touch.pos):
            return
        self.animate_button_click(instance)
        
        if "register" in [s.name for s in self.screen_manager.screens]:
            register_screen = self.screen_manager.get_screen("register")
            if hasattr(register_screen, "main_content"):
                register_screen.main_content.on_leave()

        self.screen_manager.current = "main"
        if "main" in [s.name for s in self.screen_manager.screens]:
            main_screen = self.screen_manager.get_screen("main")
            if hasattr(main_screen, "main_content"):
                main_screen.main_content.on_enter()

    def on_register_logo_pressed(self, instance, touch):
        if not instance.collide_point(*touch.pos):
            return
        self.animate_button_click(instance)
        
        if "main" in [s.name for s in self.screen_manager.screens]:
            main_screen = self.screen_manager.get_screen("main")
            if hasattr(main_screen, "main_content"):
                main_screen.main_content.on_leave()

        if "register" in [s.name for s in self.screen_manager.screens]:
            self.screen_manager.current = "register"
            register_screen = self.screen_manager.get_screen("register")
            if hasattr(register_screen, "main_content"):
                register_screen.main_content.on_enter()

    def delete_files(self, instance, touch):
        if not self.remove_logo.collide_point(*touch.pos):
            return
        
        self.animate_button_click(instance)
        
        # Load data karyawan dari API
        karyawan_list = get_karyawan_from_api()
        
        if not karyawan_list:
            dialog = MDDialog(
                title="Info",
                text="Tidak ada data karyawan",
                buttons=[MDFlatButton(text="OK", on_release=lambda x: dialog.dismiss())]
            )
            dialog.open()
            return

        self.nrp_checkboxes = {}

        content = MDBoxLayout(
            orientation="vertical", 
            spacing=10, 
            size_hint_y=None, 
            height=dp(600), 
            md_bg_color=(0.945, 0.960, 1, 1)
        )

        search_row = BoxLayout(orientation="horizontal", size_hint_y=None, height=dp(60), spacing=10)

        self.search_field = MDTextField(
            hint_text="Cari NRP/Nama...",
            size_hint_x=0.8,
            font_size=dp(20)
        )

        search_button = MDRaisedButton(
            text="Cari",
            size_hint_x=0.2,
            font_name="Poppins-Bold" if "Poppins-Bold" in LabelBase._fonts else "Roboto",
            md_bg_color=(0.631, 0.694, 0.909, 1),
            on_release=lambda x: self.filter_nrp_checkboxes(self.search_field, self.search_field.text)
        )

        search_row.add_widget(self.search_field)
        search_row.add_widget(search_button)
        content.add_widget(search_row)

        self.nrp_list_layout = BoxLayout(orientation='vertical', spacing=20, size_hint_y=None)
        self.nrp_list_layout.bind(minimum_height=self.nrp_list_layout.setter('height'))

        scroll = ScrollView(size_hint=(1, None), size=(dp(600), dp(500)), do_scroll_x=False)
        scroll.add_widget(self.nrp_list_layout)
        content.add_widget(scroll)

        for karyawan in karyawan_list:
            nrp = karyawan.get('nrp')
            name = karyawan.get('nama', nrp)
            status_wajah = karyawan.get('status_wajah', 'Belum')
            
            card = MDCard(
                orientation='horizontal',
                size_hint_y=None,
                size_hint_x=None,
                width=dp(800),
                height=dp(70),
                padding=30,
                md_bg_color=(0.901, 0.925, 1, 1),
                shadow_softness=1,
                shadow_offset=(1, 1),
                elevation=1,
                pos_hint={"center_x": 0.5},
                radius=[25, 25, 25, 25],
            )

            label = MDLabel(
                text=f"{nrp} - {name} [{status_wajah}]", 
                size_hint_x=0.8, 
                bold=True, 
                font_style="H6", 
                valign="middle", 
                font_name="Poppins-Regular" if "Poppins-Regular" in LabelBase._fonts else "Roboto"
            )
            checkbox = MDCheckbox(
                size_hint_x=0.2, 
                color_active=(0, 0, 0, 1), 
                color_inactive=(0.7, 0.7, 0.7, 1)
            )
            card.add_widget(label)
            card.add_widget(checkbox)
            self.nrp_list_layout.add_widget(card)
            self.nrp_checkboxes[nrp] = (card, checkbox)

        self.dialog_hapus_nrp = MDDialog(
            title="Pilih NRP untuk dihapus",
            type="custom",
            content_cls=content,
            buttons=[
                MDFlatButton(
                    text="Batal", 
                    font_name="Poppins-Bold" if "Poppins-Bold" in LabelBase._fonts else "Roboto", 
                    on_release=lambda x: self.dialog_hapus_nrp.dismiss()
                ),
                MDRaisedButton(
                    text="Hapus", 
                    md_bg_color=(0.631, 0.694, 0.909, 1), 
                    font_name="Poppins-Bold" if "Poppins-Bold" in LabelBase._fonts else "Roboto", 
                    on_release=self.delete_selected_nrp
                )
            ],
            size_hint=(0.8, None),
            height=dp(900),
            md_bg_color=(0.945, 0.960, 1, 1),
            auto_dismiss=False
        )
        self.dialog_hapus_nrp.open()
        self.vkeyboard = None
        self.search_field.bind(focus=self.show_keyboard)
        
    def filter_nrp_checkboxes(self, instance, value):
        value = value.lower()
        self.nrp_list_layout.clear_widgets()

        for nrp, (layout, checkbox) in self.nrp_checkboxes.items():
            if value in nrp.lower():
                self.nrp_list_layout.add_widget(layout)

    def delete_selected_nrp(self, instance):
        selected_nrp = [nrp for nrp, (_, cb) in self.nrp_checkboxes.items() if cb.active]

        if not selected_nrp:
            return

        # Hapus dari API
        result = delete_karyawan_from_api(selected_nrp)
        
        # Reload data
        thread = threading.Thread(target=load_face_data_from_api)
        thread.start()

        self.dialog_hapus_nrp.dismiss()

        if result:
            dialog = MDDialog(
                title="Berhasil",
                text=f"{len(selected_nrp)} data berhasil dihapus.",
                buttons=[MDFlatButton(text="OK", on_release=lambda x: dialog.dismiss())]
            )
        else:
            dialog = MDDialog(
                title="Gagal",
                text="Gagal menghapus data.",
                buttons=[MDFlatButton(text="OK", on_release=lambda x: dialog.dismiss())]
            )
        dialog.open()

    def show_keyboard(self, instance, value):
        if value:
            if self.vkeyboard is None:
                self.vkeyboard = VKeyboard()
                self.vkeyboard.size_hint = (1, None)
                self.vkeyboard.height = Window.height * 0.3
                self.vkeyboard.pos = (0, 0)
                self.vkeyboard.bind(on_textinput=self.on_textinput)
                self.vkeyboard.bind(on_key_down=self.on_key_down)
                Window.add_widget(self.vkeyboard)
        else:
            if self.vkeyboard:
                Window.remove_widget(self.vkeyboard)
                self.vkeyboard = None
                
    def on_textinput(self, keyboard, text):
        active_field = None
        if self.search_field.focus:
            active_field = self.search_field
        if active_field:
            active_field.text += text

    def on_key_down(self, keyboard, keycode, text, modifiers):
        active_field = None
        if self.search_field.focus:
            active_field = self.search_field

        if active_field:
            if keycode == "backspace":
                active_field.text = active_field.text[:-1]
            elif keycode == "enter":
                active_field.focus = False
            elif keycode == "escape":
                active_field.focus = False

# ==================== MAIN CONTENT (DIPERBAIKI) ====================
class MainContent(BoxLayout):
    def __init__(self, screen_manager, **kwargs):
        global manual_lat, manual_lon
        
        super().__init__(**kwargs)
        self.screen_manager = screen_manager
        self.absensi_tercatat = {}
        self.orientation = 'vertical'
        self.size_hint = (0.8, 1)
        self.last_recognition_time = 0
        self.recognition_interval = 0.5
        self.recognition_thread_running = False
        
        # Lokasi manual
        self.manual_lat = manual_lat
        self.manual_lon = manual_lon
        
        # Schedule
        Clock.schedule_interval(self.hapus_file_capture, 600)
        Clock.schedule_interval(self.hapus_file_cache, 300)
        Clock.schedule_interval(self.update_time, 1)
        Clock.schedule_interval(self.check_internet_connection, 10)
        Clock.schedule_interval(self.check_new_day, 60)
        
        # Set locale
        try:
            locale.setlocale(locale.LC_TIME, "id_ID.UTF-8")
        except:
            try:
                locale.setlocale(locale.LC_TIME, "id_ID")
            except:
                pass
        
        with self.canvas.before:
            Color(1, 1, 1, 1)
            self.rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self.update_rect, pos=self.update_rect)

        # ========== UI LAYOUT ==========
        root_layout = BoxLayout(orientation="vertical", spacing=10, padding=[10, 10, 10, 10])
        self.root_layout = root_layout

        # Header
        header_layout = BoxLayout(orientation="horizontal", size_hint=(1, None), height=dp(80), padding=[10, 0, 10, 0])
        self.header_title = Label(
            text="Face Recognition",
            color=(0.18, 0.30, 0.49, 1),
            halign="left",
            size_hint_x=None,
            width=dp(400),
            font_size=sp(38),
            font_name="Poppins-Bold" if "Poppins-Bold" in LabelBase._fonts else "Roboto"
        )
        self.location_label = Label(
            text="Memuat lokasi...",
            color=(0.4, 0.4, 0.4, 1),
            halign="right",
            size_hint_x=None,
            width=dp(400),
            font_size=sp(20),
            font_name="Poppins-Regular" if "Poppins-Regular" in LabelBase._fonts else "Roboto"
        )
        self.location_label.bind(on_touch_down=self.open_map_page)
        header_layout.add_widget(self.header_title)
        header_layout.add_widget(self.location_label)
        root_layout.add_widget(header_layout)

        # Kamera
        self.camera_layout = RelativeLayout(size_hint=(1, 1), pos_hint={"center_x": 0.5, "top": 1})
        self.camera_card = MDCard(
            size_hint=(1, 1), 
            style="elevated", 
            radius=[dp(16), dp(16), dp(16), dp(16)], 
            md_bg_color=(1, 1, 1, 1), 
            padding=0
        )
        self.camera_display = Image(size_hint=(1, 1), allow_stretch=True, keep_ratio=False)
        self.camera_card.add_widget(self.camera_display)
        self.camera_layout.add_widget(self.camera_card)

        # Info atas (tanggal & jam)
        self.top_info_container = MDBoxLayout(
            size_hint=(None, None),
            size=(dp(460), dp(48)),
            pos_hint={"center_x": 0.5, "top": 0.98},
            md_bg_color=(1, 1, 1, 0.95),
            radius=[dp(24)] * 4,
            padding=[10, 6, 10, 6],
            spacing=10
        )
        
        self.date_chip = MDBoxLayout(
            size_hint=(None, 1), 
            width=dp(220), 
            md_bg_color=(0.95, 0.95, 0.95, 1), 
            radius=[dp(18)] * 4, 
            padding=[12, 0, 12, 0]
        )
        self.date_label = Label(
            text=self.get_today_date(), 
            color=(0.2, 0.2, 0.2, 1), 
            halign="center", 
            font_size="18sp", 
            font_name="Poppins-Medium" if "Poppins-Medium" in LabelBase._fonts else "Roboto"
        )
        self.date_chip.add_widget(self.date_label)

        self.time_chip = MDBoxLayout(
            size_hint=(None, 1), 
            width=dp(160), 
            md_bg_color=(0.95, 0.95, 0.95, 1), 
            radius=[dp(18)] * 4, 
            padding=[12, 0, 12, 0]
        )
        self.time_label = Label(
            text=self.get_current_time(), 
            color=(0.2, 0.2, 0.2, 1), 
            halign="center", 
            font_size="18sp", 
            font_name="Poppins-Medium" if "Poppins-Medium" in LabelBase._fonts else "Roboto"
        )
        self.time_chip.add_widget(self.time_label)

        self.top_info_container.add_widget(self.date_chip)
        self.top_info_container.add_widget(self.time_chip)
        self.camera_layout.add_widget(self.top_info_container)

        # Instruction
        self.instruction_container = MDBoxLayout(
            size_hint=(None, None),
            size=(dp(520), dp(44)),
            pos_hint={"center_x": 0.5, "y": 0.03},
            md_bg_color=(0, 0, 0, 0.65),
            radius=[dp(10)] * 4,
            padding=[14, 6, 14, 6]
        )
        self.instruction_main = Label(
            text="Fokuskan Wajah Anda Pada Layar",
            color=(1, 1, 1, 1),
            halign="center",
            font_name="Poppins-Medium" if "Poppins-Medium" in LabelBase._fonts else "Roboto",
            font_size="18sp"
        )
        self.instruction_container.add_widget(self.instruction_main)
        self.camera_layout.add_widget(self.instruction_container)

        root_layout.add_widget(self.camera_layout)

        # Notifikasi bar
        self.active_notif_bar = None

        # Filter
        self.add_widget(root_layout)
        filter_layout = BoxLayout(orientation="horizontal", size_hint_y=None, height=dp(84), padding=(10, 10), spacing=10)
        
        self.name_filter = MDTextField(
            hint_text="Cari Nama...",
            size_hint_x=0.5,
            font_size="20sp",
            font_name="Poppins-Regular" if "Poppins-Regular" in LabelBase._fonts else "Roboto",
            on_text_validate=self.filter_table
        )
        
        self.time_filter_btn = MDRaisedButton(
            text="Semua",
            size_hint_x=None,
            width=dp(140),
            md_bg_color=(0.631, 0.694, 0.909, 1),
            text_color=(0.203, 0.2, 0.2, 1),
            font_size="18sp",
            font_name="Poppins-Medium" if "Poppins-Medium" in LabelBase._fonts else "Roboto",
            on_release=self.open_time_menu
        )
        
        self.time_menu = MDDropdownMenu(
            caller=self.time_filter_btn,
            items=[
                {"viewclass": "OneLineListItem", "text": "Semua", "on_release": lambda x="Semua": self.set_time_filter(x)},
                {"viewclass": "OneLineListItem", "text": "07:30 - 12:00", "on_release": lambda x="07:30 - 12:00": self.set_time_filter(x)},
                {"viewclass": "OneLineListItem", "text": "12:00 - 18:30", "on_release": lambda x="12:00 - 18:30": self.set_time_filter(x)}
            ],
            width_mult=3
        )
        
        filter_layout.add_widget(self.name_filter)
        filter_layout.add_widget(self.time_filter_btn)
        self.add_widget(filter_layout)

        # Tabel
        self.original_data = []
        self.filtered_data = []
        self.load_table_data()
        
        # Keyboard
        self.vkeyboard = None
        self.name_filter.bind(focus=self.show_keyboard)
        self.lat_input = None
        self.lon_input = None
        
        # Update lokasi
        Clock.schedule_once(lambda dt: self.update_location(lat=self.manual_lat, lon=self.manual_lon), 2)

    # ========== FUNGSI UTAMA ==========
    def get_current_time(self):
        return datetime.now().strftime("%H:%M:%S")

    def update_time(self, dt):
        if hasattr(self, "time_label"):
            self.time_label.text = self.get_current_time()

    def get_today_date(self):
        today = datetime.now()
        days = {
            'Monday': 'Senin', 'Tuesday': 'Selasa', 'Wednesday': 'Rabu',
            'Thursday': 'Kamis', 'Friday': 'Jumat', 'Saturday': 'Sabtu',
            'Sunday': 'Minggu'
        }
        months = {
            'January': 'Januari', 'February': 'Februari', 'March': 'Maret',
            'April': 'April', 'May': 'Mei', 'June': 'Juni',
            'July': 'Juli', 'August': 'Agustus', 'September': 'September',
            'October': 'Oktober', 'November': 'November', 'December': 'Desember'
        }
        
        day_name = days.get(today.strftime('%A'), today.strftime('%A'))
        month_name = months.get(today.strftime('%B'), today.strftime('%B'))
        
        return f"{day_name}, {today.day} {month_name} {today.year}"

    def update_rect(self, *args):
        self.rect.size = self.size
        self.rect.pos = self.pos

    def on_enter(self):
        """Aktifkan kamera"""
        thread = threading.Thread(target=load_face_data_from_api)
        thread.start()

        self.cap = CameraSingleton.get_instance(screen_name="MainContent")
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        Clock.schedule_interval(self.update_camera, 1.0 / 30.0)

    def on_leave(self):
        Clock.unschedule(self.update_camera)
        CameraSingleton.release(screen_name="MainContent")

    def update_camera(self, dt):
        global recognized_faces, face_data
        
        if not hasattr(self, 'cap') or self.cap is None:
            return
            
        success, img = self.cap.read()
        if not success:
            return
            
        img = cv2.flip(img, 1)
        frame_width = img.shape[1]
        img_for_detection = cv2.flip(img, 1)
        
        # YOLO detection
        results = model(img, stream=True, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                conf = float(box.conf[0])  
                cls = int(box.cls[0])

                if conf > confidence:
                    color = (0, 255, 0) if classNames[cls] == 'real' else (0, 0, 255)
                    cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)

                    if classNames[cls] == 'real':
                        now = time.time()
                        if not self.recognition_thread_running and (now - self.last_recognition_time > self.recognition_interval):
                            self.recognition_thread_running = True
                            self.last_recognition_time = now

                            def recognition_done(*a):
                                self.recognition_thread_running = False

                            def thread_func():
                                recognize_face(img_for_detection.copy(), self)
                                Clock.schedule_once(recognition_done)
                            threading.Thread(target=thread_func).start()

                        current_time = time.time()
                        recognized_faces = {nrp: (x, y, t) for nrp, (x, y, t) in recognized_faces.items() if current_time - t < 2}

                        valid_faces = {}
                        for nrp, (x1, y1, _) in recognized_faces.items():
                            if nrp != "Unknown":
                                valid_faces[nrp] = (x1, y1, _)

                        if not valid_faces:
                            valid_faces = recognized_faces

                        for nrp, (x1, y1, _) in valid_faces.items():
                            if nrp == "Unknown":
                                label = "Tidak Dikenal"
                                color = (0, 0, 255)
                            else:
                                name = face_data.get(nrp, {}).get('name', nrp)
                                label = f"{name}"
                                color = (0, 255, 0)

                            mirrored_text_x = frame_width - x1
                            cvzone.putTextRect(
                                img,
                                label,
                                (max(0, mirrored_text_x), max(100, y1)),
                                scale=1.5,
                                thickness=2,
                                colorR=color
                            )

        # Crop untuk portrait
        h, w, _ = img.shape
        x_center, y_center = w // 2, h // 2
        crop_w, crop_h = 960, 1280
        x1 = max(0, x_center - crop_w // 2)
        x2 = min(w, x_center + crop_w // 2)
        y1 = max(0, y_center - crop_h // 2)
        y2 = min(h, y_center + crop_h // 2)
        img = img[y1:y2, x1:x2]

        # Convert to texture
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 0)
        texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='rgb')
        texture.blit_buffer(img.tobytes(), colorfmt='rgb', bufferfmt='ubyte')

        self.camera_display.texture = texture

    # ===== METHOD SEND ABSENSI YANG DIPERBAIKI =====
    def send_absensi(self, nrp, image_path_local):
        """Mengirim data absensi ke API"""
        global manual_lat, manual_lon
        
        print(f"📤 Mengirim absensi untuk NRP: {nrp}")
        
        # Kirim ke API
        response = send_absensi_to_api(
            nrp=nrp,
            image_path=image_path_local,
            latitude=self.manual_lat,
            longitude=self.manual_lon
        )
        
        print(f"📥 Response dari API: {response}")
        
        if response and response.get('status') == 200:
            data = response.get('data', {})
            message = response.get('message', 'Absensi Berhasil')
            
            # Ambil data dari response
            name = data.get('name', nrp)
            waktu_full = data.get('waktu', '')
            
            # Format waktu untuk tampilan (HH:MM:SS)
            try:
                if waktu_full:
                    waktu_obj = datetime.strptime(waktu_full, "%Y-%m-%d %H:%M:%S")
                    waktu_display = waktu_obj.strftime("%H:%M:%S")
                else:
                    waktu_display = datetime.now().strftime("%H:%M:%S")
            except:
                waktu_display = datetime.now().strftime("%H:%M:%S")
            
            # Tentukan status yang akan ditampilkan
            if "Check In" in message:
                status_display = "Check In"
                self.show_checkin_success_dialog()
            elif "Check Out" in message:
                status_display = "Check Out"
                self.show_checkout_success_dialog()
            else:
                status_display = message
            
            print(f"✅ Akan ditambahkan ke tabel: {name} - {status_display} - {waktu_display}")
            
            # Tambahkan ke tabel
            self.add_to_table(name, status_display, waktu_display)
            self.show_absen_notif(name, status_display, waktu_display)
            
            print(f"✅ Absensi berhasil: {name} ({nrp}) - {status_display}")
        else:
            error_msg = response.get('message', 'Unknown error') if response else 'Tidak ada response'
            print(f"❌ Absensi gagal: {nrp} - {error_msg}")

    def show_absen_notif(self, name, status, waktu):
        if self.active_notif_bar and self.active_notif_bar.parent:
            self.root_layout.remove_widget(self.active_notif_bar)
            self.active_notif_bar = None

        notif_bar = MDBoxLayout(
            size_hint=(1, None),
            height=dp(52),
            md_bg_color=(0.18, 0.30, 0.49, 1),
            padding=[10, 8, 10, 8],
            spacing=10
        )
        
        status_chip = MDBoxLayout(
            size_hint=(None, 1),
            width=dp(220),
            md_bg_color=(0.36, 0.75, 0.45, 1),
            radius=[dp(18)] * 4,
            padding=[12, 0, 12, 0]
        )
        status_label = Label(
            text=f"{status}",
            color=(1, 1, 1, 1),
            halign="center",
            font_name="Poppins-Medium" if "Poppins-Medium" in LabelBase._fonts else "Roboto",
            font_size="18sp"
        )
        status_chip.add_widget(status_label)

        filler = Label(
            text=f"{name} - {waktu}", 
            theme_text_color="Custom", 
            color=(1, 1, 1, 0.85), 
            halign="left", 
            font_size="18sp", 
            font_name="Poppins-Regular" if "Poppins-Regular" in LabelBase._fonts else "Roboto"
        )
        close_btn = MDRaisedButton(
            text="Tutup",
            md_bg_color=(0.12, 0.20, 0.36, 1),
            text_color=(1, 1, 1, 1),
            on_release=lambda x: self._dismiss_notif_bar()
        )

        notif_bar.add_widget(status_chip)
        notif_bar.add_widget(filler)
        notif_bar.add_widget(close_btn)

        self.active_notif_bar = notif_bar
        self.root_layout.add_widget(self.active_notif_bar, index=1)
        Clock.schedule_once(lambda dt: self._dismiss_notif_bar(), 2.5)

    def _dismiss_notif_bar(self):
        if self.active_notif_bar and self.active_notif_bar.parent:
            try:
                self.root_layout.remove_widget(self.active_notif_bar)
            except:
                pass
        self.active_notif_bar = None

    def show_checkin_success_dialog(self):
        content = MDBoxLayout(
            orientation="vertical",
            spacing=dp(20),
            padding=dp(30),
            size_hint_y=None,
            height=dp(300)
        )
        content.bind(minimum_height=content.setter('height'))

        icon = Image(
            source="assets/icon_checklist.png",
            size_hint=(None, None),
            size=(dp(100), dp(100)),
            pos_hint={"center_x": 0.5}
        )

        label = MDLabel(
            text="Check In Berhasil",
            halign="center",
            font_style="H5",
            theme_text_color="Primary",
            font_name="Poppins-Bold" if "Poppins-Bold" in LabelBase._fonts else "Roboto",
            size_hint_y=None,
            height=dp(40)
        )

        content.add_widget(icon)
        content.add_widget(label)

        dialog = MDDialog(
            type="custom",
            content_cls=content,
            size_hint=(0.8, None),
            height=dp(300),
            auto_dismiss=True
        )
        dialog.open()
        Clock.schedule_once(lambda dt: dialog.dismiss(), 2.0)

    def show_checkout_success_dialog(self):
        content = MDBoxLayout(
            orientation="vertical",
            spacing=dp(20),
            padding=dp(30),
            size_hint_y=None,
            height=dp(300)
        )
        content.bind(minimum_height=content.setter('height'))

        icon = Image(
            source="assets/icon_checklist.png",
            size_hint=(None, None),
            size=(dp(100), dp(100)),
            pos_hint={"center_x": 0.5}
        )

        label = MDLabel(
            text="Check Out Berhasil",
            halign="center",
            font_style="H5",
            theme_text_color="Primary",
            font_name="Poppins-Bold" if "Poppins-Bold" in LabelBase._fonts else "Roboto",
            size_hint_y=None,
            height=dp(40)
        )

        content.add_widget(icon)
        content.add_widget(label)

        dialog = MDDialog(
            type="custom",
            content_cls=content,
            size_hint=(0.8, None),
            height=dp(300),
            auto_dismiss=True
        )
        dialog.open()
        Clock.schedule_once(lambda dt: dialog.dismiss(), 2.0)

    def show_no_internet_dialog(self):
        if hasattr(self, 'no_internet_dialog') and self.no_internet_dialog:
            return
        
        no_internet_container = MDBoxLayout(
            size_hint=(None, None),
            size=(dp(500), dp(80)),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            md_bg_color=(0.921, 0.364, 0.380, 1),
            radius=[dp(25), dp(25), dp(25), dp(25)],
            padding=[20, 10, 20, 10]
        )
        no_internet_label = Label(
            text="[size=25]Jaringan internet anda terputus, silahkan periksa kembali.[/size]", 
            font_name="Poppins-Medium" if "Poppins-Medium" in LabelBase._fonts else "Roboto", 
            halign="center", 
            pos_hint={"center_x":0.5, "center_y":0.5}, 
            markup=True
        )
        no_internet_container.add_widget(no_internet_label)

        no_internet_content = MDBoxLayout(orientation='vertical', spacing=10, padding=30, md_bg_color=(0, 0, 0, 0))
        no_internet_content.add_widget(Image(
            source="assets/no_internet_icon.png", 
            size_hint=(1, None), 
            height=250, 
            pos_hint={"center_x":0.5, "y":0.4}
        ))
        no_internet_content.add_widget(no_internet_container)
        
        self.no_internet_dialog = MDDialog(
            type="custom",
            height=50,
            radius=[dp(20), dp(20), dp(20), dp(20)],
            content_cls=no_internet_content,
        )
        self.no_internet_dialog.open()

    def check_internet_connection(self, dt):
        if not is_internet_available():
            if not hasattr(self, 'no_internet_dialog') or not self.no_internet_dialog:
                self.show_no_internet_dialog()
        else:
            if hasattr(self, 'no_internet_dialog') and self.no_internet_dialog:
                self.no_internet_dialog.dismiss()
                self.no_internet_dialog = None

    def update_location(self, **kwargs):
        global manual_lat, manual_lon
        
        lat = kwargs.get('lat', manual_lat)
        lon = kwargs.get('lon', manual_lon)
        
        manual_lat = lat
        manual_lon = lon
        self.manual_lat = lat
        self.manual_lon = lon

        address = self.get_address_from_lat_lon(lat, lon)
        self.location_label.text = address
        self.location_label.texture_update()

    def get_address_from_lat_lon(self, lat, lon):
        try:
            url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=18&addressdetails=1"
            headers = {"User-Agent": "AttendanceHRIS/1.0"}
            response = requests.get(url, headers=headers, timeout=5)
            data = response.json()

            if "address" in data:
                address_components = data["address"]
                desa = address_components.get("village", "") or address_components.get("town", "") or address_components.get("city", "")
                kota = address_components.get("city", "") or address_components.get("county", "")
                provinsi = address_components.get("state", "")

                if lat == -6.866641 and lon == 107.5347632:
                    desa = "Cipageran"

                address_parts = [part for part in [desa, kota, provinsi] if part]
                return ", ".join(address_parts) if address_parts else "Alamat tidak ditemukan"
            return "Alamat tidak ditemukan"
        except:
            return "Lokasi Tidak Diketahui"

    def open_map_page(self, instance, touch):
        global manual_lat, manual_lon
        
        if self.location_label.collide_point(*touch.pos):
            self.clear_widgets()

            main_layout = BoxLayout(orientation="vertical", spacing=20, padding=10, size_hint=(1, 1))

            map_layout = BoxLayout(size_hint=(1, 3), pos_hint={"center_x":0.5})
            self.mapview = MapView(zoom=15, lat=manual_lat, lon=manual_lon, size_hint=(1, 1))
            map_layout.add_widget(self.mapview)

            self.marker = MapMarker(lat=manual_lat, lon=manual_lon)
            self.mapview.add_widget(self.marker)

            input_layout = BoxLayout(orientation="vertical", size_hint=(1, 0.3), width=dp(1000), spacing=20)

            self.lat_input_label = MDBoxLayout(
                size_hint=(0.9, None),
                size=(dp(1000), dp(70)),
                pos_hint={"center_x": 0.5},
                md_bg_color=(0.905, 0.905, 0.909, 1),
                radius=[dp(10), dp(10), dp(10), dp(10)],
                padding=10
            )

            self.lat_input = MDTextField(
                hint_text="Latitude",
                text=str(manual_lat),
                size_hint_x=1,
                font_size=dp(20),
            )
            self.lat_input_label.add_widget(self.lat_input)

            self.lon_input_label = MDBoxLayout(
                size_hint=(0.9, None),
                size=(dp(1000), dp(70)),
                pos_hint={"center_x": 0.5},
                md_bg_color=(0.905, 0.905, 0.909, 1),
                radius=[dp(10), dp(10), dp(10), dp(10)],
                padding=10
            )

            self.lon_input = MDTextField(
                hint_text="Longitude",
                text=str(manual_lon),
                size_hint_x=1,
                font_size=dp(20),
            )
            self.lon_input_label.add_widget(self.lon_input)

            self.lat_input.bind(focus=self.show_keyboard)
            self.lon_input.bind(focus=self.show_keyboard)

            input_layout.add_widget(self.lat_input_label)
            input_layout.add_widget(self.lon_input_label)

            button_layout = BoxLayout(size_hint=(1, 0.1), pos_hint={"y": 0}, spacing=35)

            back_button = MDRaisedButton(
                text="Kembali",
                size_hint_x=0.5,
                md_bg_color=(0.203, 0.2, 0.2, 1),
                font_name="Poppins-Medium" if "Poppins-Medium" in LabelBase._fonts else "Roboto",
                font_size="20sp",
                on_release=self.back_to_main
            )

            update_button = MDRaisedButton(
                text="Perbarui",
                size_hint_x=0.5,
                text_color=(0, 0, 0, 1),
                md_bg_color=(0.631, 0.694, 0.909, 1),
                font_name="Poppins-Medium" if "Poppins-Medium" in LabelBase._fonts else "Roboto",
                font_size="20sp",
                on_release=self.update_marker_location
            )

            button_layout.add_widget(back_button)
            button_layout.add_widget(update_button)

            main_layout.add_widget(map_layout)
            main_layout.add_widget(input_layout)
            main_layout.add_widget(button_layout)

            self.add_widget(main_layout)

    def show_keyboard(self, instance, value):
        if value:
            if self.vkeyboard is None:
                self.vkeyboard = VKeyboard()
                self.vkeyboard.size_hint = (1, None)
                self.vkeyboard.height = Window.height * 0.30
                self.vkeyboard.bind(on_textinput=self.on_textinput)
                self.vkeyboard.bind(on_key_down=self.on_key_down)
                self.add_widget(self.vkeyboard)
        else:
            if self.vkeyboard:
                self.remove_widget(self.vkeyboard)
                self.vkeyboard = None

    def on_textinput(self, keyboard, text):
        active_field = None
        if self.lat_input and self.lat_input.focus:
            active_field = self.lat_input
        elif self.lon_input and self.lon_input.focus:
            active_field = self.lon_input
        elif self.name_filter.focus:
            active_field = self.name_filter

        if active_field:
            active_field.text += text

    def on_key_down(self, keyboard, keycode, text, modifiers):
        active_field = None
        if self.lat_input and self.lat_input.focus:
            active_field = self.lat_input
        elif self.lon_input and self.lon_input.focus:
            active_field = self.lon_input
        elif self.name_filter.focus:
            active_field = self.name_filter

        if active_field:
            if keycode == "backspace":
                active_field.text = active_field.text[:-1]
            elif keycode == "enter":
                active_field.focus = False
            elif keycode == "escape":
                active_field.focus = False

    def update_marker_location(self, instance):
        global manual_lat, manual_lon
        
        try:
            new_lat = float(self.lat_input.text)
            new_lon = float(self.lon_input.text)

            manual_lat = new_lat
            manual_lon = new_lon
            self.manual_lat = new_lat
            self.manual_lon = new_lon

            self.mapview.center_on(new_lat, new_lon)

            if hasattr(self, 'marker'):
                self.mapview.remove_widget(self.marker)

            self.marker = MapMarker(lat=new_lat, lon=new_lon)
            self.mapview.add_widget(self.marker)

            self.location_label.text = self.get_address_from_lat_lon(new_lat, new_lon)
            
            self.show_update_success_dialog()

        except ValueError:
            pass

    def show_update_success_dialog(self):
        label_container = MDBoxLayout(
            size_hint=(None, None),
            size=(dp(500), dp(60)),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            md_bg_color=(0.564, 0.874, 0.654, 1),
            radius=[dp(25), dp(25), dp(25), dp(25)],
            padding=[20, 10, 20, 10]
        )
        success_label = Label(
            text="Lokasi berhasil diperbarui", 
            font_name="Poppins-Medium" if "Poppins-Medium" in LabelBase._fonts else "Roboto", 
            halign="center", 
            pos_hint={"center_x":0.5, "center_y":0.5}
        )
        label_container.add_widget(success_label)

        content = MDBoxLayout(orientation='vertical', spacing=10, padding=30, md_bg_color=(0, 0, 0, 0))
        content.add_widget(Image(
            source="assets/complete_icon.png", 
            size_hint=(1, None), 
            height=250, 
            pos_hint={"center_x":0.5, "y":0.4}
        ))
        content.add_widget(label_container)
        
        dialog = MDDialog(
            type="custom",
            height=50,
            radius=[dp(20), dp(20), dp(20), dp(20)],
            content_cls=content,
            buttons=[
                MDRaisedButton(
                    text="Tutup",
                    md_bg_color=(0.631, 0.694, 0.909, 1),
                    text_color=(0.203, 0.2, 0.2, 1),
                    on_release=lambda x: dialog.dismiss()
                )
            ]
        )
        dialog.open()
        Clock.schedule_once(lambda x: dialog.dismiss(), 10)

    def back_to_main(self, instance):
        self.clear_widgets()
        self.__init__(self.screen_manager)
        self.on_enter()
        self.update_location(lat=manual_lat, lon=manual_lon)

    def update_pagination_font(self, *args):
        if hasattr(self, 'table'):
            pagination = self.table.pagination
            if pagination:
                if "label_rows_per_page" in pagination.ids:
                    pagination.ids.label_rows_per_page.font_size = "20sp"
                    pagination.ids.label_rows_per_page.font_name = "Poppins-Regular" if "Poppins-Regular" in LabelBase._fonts else "Roboto"
                if "drop_item" in pagination.ids:
                    pagination.ids.drop_item.font_size = "20sp"
                    pagination.ids.drop_item.font_name = "Poppins-Regular" if "Poppins-Regular" in LabelBase._fonts else "Roboto"

    # ===== METHOD TABEL YANG DIPERBAIKI =====
    def create_table(self):
        """Membuat tabel data absensi"""
        if hasattr(self, 'table'):
            self.remove_widget(self.table)
            # Hapus referensi tabel
            delattr(self, 'table')

        # Pastikan data dalam format yang benar
        row_data = []
        for item in self.filtered_data:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                row_data.append((
                    str(item[0]),  # Nama
                    str(item[1]),  # Status
                    str(item[2])   # Waktu
                ))
        
        print(f"📊 Membuat tabel dengan {len(row_data)} baris: {row_data}")

        self.table = MDDataTable(
            size_hint=(None, None),
            width=dp(840),
            height=dp(640),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            use_pagination=True,
            column_data=[
                ("[size=20][b]Nama[/b][/size]", dp(50)),
                ("[size=20][b]Status[/b][/size]", dp(70)),
                ("[size=20][b]Waktu[/b][/size]", dp(50))
            ],
            row_data=row_data
        )
        
        self.add_widget(self.table)
        Clock.schedule_once(lambda dt: self.update_pagination_font(), 0.1)

    def add_to_table(self, name, status, waktu):
        """Menambahkan data absensi ke tabel"""
        # Validasi data
        if not name or not status or not waktu:
            print(f"❌ Data tidak valid: {name}, {status}, {waktu}")
            return
        
        # Buat entry baru
        new_entry = (str(name), str(status), str(waktu))
        
        # Tambahkan ke awal list
        self.original_data.insert(0, new_entry)
        self.filtered_data = self.original_data.copy()
        
        print(f"✅ Data ditambahkan ke tabel: {new_entry}")
        print(f"📊 Total data dalam tabel: {len(self.original_data)}")
        
        # Perbarui tabel
        self.create_table()
        self.save_table_data()

    def save_table_data(self):
        """Menyimpan data tabel ke file JSON"""
        data_to_save = {
            "tanggal": datetime.now().strftime("%Y-%m-%d"),
            "absensi": self.original_data
        }
        try:
            with open("data_table.json", "w") as file:
                json.dump(data_to_save, file, indent=4)
            print(f"💾 Data tersimpan: {len(self.original_data)} entries")
        except Exception as e:
            print(f"❌ Error save table: {e}")

    def load_table_data(self):
        """Memuat data tabel dari file JSON"""
        try:
            with open("data_table.json", "r") as file:
                data = json.load(file)

            last_saved_date = data.get("tanggal", "")
            today_date = datetime.now().strftime("%Y-%m-%d")

            if last_saved_date == today_date:
                raw_data = data.get("absensi", [])
                # Konversi ke tuple jika perlu
                self.original_data = []
                for item in raw_data:
                    if isinstance(item, list):
                        self.original_data.append(tuple(item))
                    else:
                        self.original_data.append(item)
                
                self.filtered_data = self.original_data.copy()
                print(f"📂 Memuat {len(self.original_data)} data dari file")
            else:
                print("🔄 Hari baru, reset data absensi")
                self.original_data = []
                self.filtered_data = []
                self.save_table_data()

            self.create_table()
        except FileNotFoundError:
            print("📂 File data_table.json tidak ditemukan, membuat baru")
            self.original_data = []
            self.filtered_data = []
            self.create_table()
            self.save_table_data()
        except Exception as e:
            print(f"❌ Error load table data: {e}")
            self.original_data = []
            self.filtered_data = []
            self.create_table()

    def filter_table(self, instance=None):
        name_filter = self.name_filter.text.lower()
        time_filter = self.time_filter_btn.text

        self.filtered_data = [
            row for row in self.original_data
            if (name_filter in row[0].lower()) and (time_filter == "Semua" or self.is_time_in_range(row[2], time_filter))
        ]
        self.create_table()

    def open_time_menu(self, instance):
        self.time_menu.open()

    def set_time_filter(self, time_range):
        self.time_filter_btn.text = time_range
        self.time_menu.dismiss()
        self.filter_table()

    def is_time_in_range(self, time_str, time_range):
        if time_range == "Semua":
            return True

        try:
            start_time_str, end_time_str = time_range.split(" - ")
            time_obj = datetime.strptime(time_str, "%H:%M:%S").time()
            start_time_obj = datetime.strptime(start_time_str, "%H:%M").time()
            end_time_obj = datetime.strptime(end_time_str, "%H:%M").time()
            return start_time_obj <= time_obj <= end_time_obj
        except:
            return False

    def check_new_day(self, dt):
        today = datetime.now().strftime("%Y-%m-%d")
        if today != getattr(self, 'last_checked_date', ''):
            self.last_checked_date = today
            self.load_table_data()

    def hapus_file_capture(self, *args):
        now = time.time()
        try:
            for filename in os.listdir(CAPTURE_FOLDER):
                file_path = os.path.join(CAPTURE_FOLDER, filename)
                if os.path.isfile(file_path):
                    file_age = now - os.path.getmtime(file_path)
                    if file_age > 600:  # 10 menit
                        os.remove(file_path)
                        print(f"🗑 Hapus file capture: {filename}")
        except:
            pass

    def hapus_file_cache(self, *args):
        now = time.time()
        try:
            for filename in os.listdir(CACHE_FOLDER):
                file_path = os.path.join(CACHE_FOLDER, filename)
                if os.path.isfile(file_path):
                    file_age = now - os.path.getmtime(file_path)
                    if file_age > 300:  # 5 menit
                        os.remove(file_path)
                        print(f"🗑 Hapus file cache: {filename}")
        except:
            pass

# ==================== REGISTRATION ====================
class Registration(BoxLayout):
    def __init__(self, screen_manager, **kwargs):
        super().__init__(**kwargs)
        self.screen_manager = screen_manager
        self.orientation = 'vertical'
        self.size_hint = (0.8, 1)

        with self.canvas.before:
            Color(1, 1, 1, 1)
            self.rect = Rectangle(size=self.size, pos=self.pos)

        self.bind(size=self.update_rect, pos=self.update_rect)

        # ROOT LAYOUT
        root_layout = BoxLayout(orientation="vertical", spacing=10, padding=[10, 10, 10, 10])
        self.root_layout = root_layout

        # HEADER
        header_layout = BoxLayout(orientation="horizontal", size_hint=(1, None), height=dp(80), padding=[10, 0, 10, 0])
        self.header_title = Label(
            text="Registration Form",
            color=(0.18, 0.30, 0.49, 1),
            halign="left",
            size_hint_x=None,
            width=dp(400),
            font_size=sp(38),
            font_name="Poppins-Bold" if "Poppins-Bold" in LabelBase._fonts else "Roboto"
        )
        self.subtitle_label = Label(
            text="Daftarkan Wajah Anda",
            color=(0.4, 0.4, 0.4, 1),
            halign="right",
            size_hint_x=None,
            width=dp(400),
            font_size=sp(20),
            font_name="Poppins-Regular" if "Poppins-Regular" in LabelBase._fonts else "Roboto"
        )
        header_layout.add_widget(self.header_title)
        header_layout.add_widget(self.subtitle_label)
        root_layout.add_widget(header_layout)

        # KAMERA
        self.camera_layout = RelativeLayout(size_hint=(1, 1), pos_hint={"center_x": 0.5, "top": 1})
        self.camera_card = MDCard(
            size_hint=(1, 1), 
            style="elevated", 
            radius=[dp(16), dp(16), dp(16), dp(16)], 
            md_bg_color=(1, 1, 1, 1), 
            padding=0
        )
        self.camera_display = Image(size_hint=(1, 1), allow_stretch=True, keep_ratio=False)
        self.camera_card.add_widget(self.camera_display)
        self.camera_layout.add_widget(self.camera_card)

        # Overlay progress
        self.top_info_container = MDBoxLayout(
            size_hint=(None, None),
            size=(dp(460), dp(48)),
            pos_hint={"center_x": 0.5, "top": 0.98},
            md_bg_color=(1, 1, 1, 0.95),
            radius=[dp(24)] * 4,
            padding=[10, 6, 10, 6],
            spacing=10
        )
        
        self.progress_chip = MDBoxLayout(
            size_hint=(None, 1),
            width=dp(220),
            md_bg_color=(0.95, 0.95, 0.95, 1),
            radius=[dp(18)] * 4,
            padding=[12, 0, 12, 0]
        )
        self.progress_label = Label(
            text="Progress: 0%",
            color=(0.2, 0.2, 0.2, 1),
            halign="center",
            font_size="18sp",
            font_name="Poppins-Medium" if "Poppins-Medium" in LabelBase._fonts else "Roboto"
        )
        self.progress_chip.add_widget(self.progress_label)

        self.status_chip = MDBoxLayout(
            size_hint=(None, 1),
            width=dp(160),
            md_bg_color=(0.95, 0.95, 0.95, 1),
            radius=[dp(18)] * 4,
            padding=[12, 0, 12, 0]
        )
        self.status_label = Label(
            text="Siap",
            color=(0.2, 0.2, 0.2, 1),
            halign="center",
            font_size="18sp",
            font_name="Poppins-Medium" if "Poppins-Medium" in LabelBase._fonts else "Roboto"
        )
        self.status_chip.add_widget(self.status_label)

        self.top_info_container.add_widget(self.progress_chip)
        self.top_info_container.add_widget(self.status_chip)
        self.camera_layout.add_widget(self.top_info_container)

        # Instruction
        self.instruction_container = MDBoxLayout(
            size_hint=(None, None),
            size=(dp(520), dp(44)),
            pos_hint={"center_x": 0.5, "y": 0.03},
            md_bg_color=(0, 0, 0, 0.65),
            radius=[dp(10)] * 4,
            padding=[14, 6, 14, 6]
        )
        self.instruction_main = Label(
            text="Fokuskan Wajah Anda Pada Layar untuk Registrasi",
            color=(1, 1, 1, 1),
            halign="center",
            font_name="Poppins-Medium" if "Poppins-Medium" in LabelBase._fonts else "Roboto",
            font_size="18sp"
        )
        self.instruction_container.add_widget(self.instruction_main)
        self.camera_layout.add_widget(self.instruction_container)

        root_layout.add_widget(self.camera_layout)

        # INPUT FORM
        self.input_layout = BoxLayout(
            orientation="horizontal", 
            size_hint=(1, None), 
            height=dp(100), 
            padding=(20, 10, 20, 10), 
            spacing=20
        )

        self.nrp_container = MDBoxLayout(orientation="vertical", size_hint=(0.6, 1), spacing=5)
        self.nrp_input = MDTextField(
            hint_text="Masukkan 10 digit NRP Anda",
            size_hint=(1, None),
            height=dp(50),
            font_size="18sp",
            font_name="Poppins-Regular" if "Poppins-Regular" in LabelBase._fonts else "Roboto",
            mode="rectangle"
        )
        self.nrp_container.add_widget(self.nrp_input)

        self.button_container = BoxLayout(orientation="horizontal", size_hint=(0.4, 1), spacing=10)

        self.cancel_button = MDRaisedButton(
            text="Batal",
            size_hint=(0.5, 1),
            md_bg_color=(0.913, 0.513, 0.525, 1),
            text_color=(1, 1, 1, 1),
            font_name="Poppins-Medium" if "Poppins-Medium" in LabelBase._fonts else "Roboto",
            on_release=self.cancel_registration
        )

        self.register_button = MDRaisedButton(
            text="Daftarkan",
            size_hint=(0.5, 1),
            md_bg_color=(0.631, 0.694, 0.909, 1),
            text_color=(1, 1, 1, 1),
            font_name="Poppins-Medium" if "Poppins-Medium" in LabelBase._fonts else "Roboto",
            on_release=self.start_face_registration
        )

        self.button_container.add_widget(self.cancel_button)
        self.button_container.add_widget(self.register_button)

        self.input_layout.add_widget(self.nrp_container)
        self.input_layout.add_widget(self.button_container)
        root_layout.add_widget(self.input_layout)

        self.add_widget(root_layout)

        self.cap = None
        self.vkeyboard = None
        self.is_registering_face = False
        self.face_encodings_list = []
        self.nrp_input.bind(focus=self.show_keyboard)
        self.update_progress_bar(0)
        self.locked_face = None
        self.last_seen_time = 0

    def update_rect(self, *args):
        self.rect.size = self.size
        self.rect.pos = self.pos

    def update_progress_bar(self, value):
        self.progress_label.text = f"Progress: {int(value)}%"
        if value == 100:
            self.status_label.text = "Selesai"
        elif value > 0:
            self.status_label.text = "Memproses"
        else:
            self.status_label.text = "Siap"

    def on_enter(self):
        self.cap = CameraSingleton.get_instance(screen_name="Registration")
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        Clock.schedule_interval(self.update_camera_reg, 1.0 / 30.0)

    def on_leave(self):
        Clock.unschedule(self.update_camera_reg)
        CameraSingleton.release(screen_name="Registration")

    def update_camera_reg(self, dt):
        if not hasattr(self, 'cap') or self.cap is None:
            return
            
        success, frame = self.cap.read()
        if not success:
            return

        frame = cv2.flip(frame, 1)
        
        # Crop frame
        h, w, _ = frame.shape
        x_center, y_center = w // 2, h // 2
        crop_w, crop_h = 960, 1280
        x1 = max(0, x_center - crop_w // 2)
        x2 = min(w, x_center + crop_w // 2)
        y1 = max(0, y_center - crop_h // 2)
        y2 = min(h, y_center + crop_h // 2)
        frame_cropped = frame[y1:y2, x1:x2]
        
        img_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)
        results = model(frame_cropped, stream=False, verbose=False)
        face_detected = False
        current_time = time.time()

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                conf = float(box.conf[0])  
                cls = int(box.cls[0])

                if conf > confidence and classNames[cls] == "real":
                    cvzone.cornerRect(frame_cropped, (x1, y1, w, h), colorC=(0, 255, 0), colorR=(0, 255, 0))
                    face_detected = True

                    if self.is_registering_face:
                        faces = app.get(img_rgb)
                        best_face = None
                        best_iou = 0
                        for face in faces:
                            fx1, fy1, fx2, fy2 = face.bbox.astype(int)
                            xx1 = max(x1, fx1)
                            yy1 = max(y1, fy1)
                            xx2 = min(x2, fx2)
                            yy2 = min(y2, fy2)
                            inter_area = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                            box_area = (x2 - x1) * (y2 - y1)
                            face_area = (fx2 - fx1) * (fy2 - fy1)
                            union_area = box_area + face_area - inter_area
                            iou = inter_area / union_area if union_area > 0 else 0
                            if iou > best_iou:
                                best_iou = iou
                                best_face = face
                                
                        if best_face is not None and best_iou > 0.1:
                            face_embedding = best_face.normed_embedding.tolist()
                            if self.locked_face is None:
                                self.locked_face = face_embedding
                                self.last_seen_time = time.time()

                            similarity = np.dot(self.locked_face, face_embedding)
                            if similarity > 0.6:
                                self.face_encodings_list.append(face_embedding)
                                progress_value = (len(self.face_encodings_list) / 15) * 100
                                self.update_progress_bar(progress_value)
                                
                                remaining = 15 - len(self.face_encodings_list)
                                self.instruction_main.text = f"Rekam wajah... {remaining} frame tersisa"

                            if len(self.face_encodings_list) >= 15:
                                self.save_face_data()
                                self.locked_face = None
                                self.last_seen_time = 0
                                return

        if self.locked_face and not face_detected:
            if (current_time - self.last_seen_time) > 3:
                self.locked_face = None
            else:
                face_detected = True

        if face_detected:
            self.last_seen_time = current_time

        # Convert to texture
        buf = cv2.flip(frame_cropped, 0).tobytes()
        texture = Texture.create(size=(frame_cropped.shape[1], frame_cropped.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.camera_display.texture = texture

    def start_face_registration(self, *args):
        nrp = self.nrp_input.text.strip()

        if not nrp:
            self.show_warning_dialog("NRP harus diisi!")
            return
            
        if len(nrp) != 10:
            self.show_warning_dialog("NRP harus terdiri dari 10 digit!")
            return

        # Cek apakah NRP ada di database
        karyawan = get_karyawan_from_api(nrp)
        
        if not karyawan:
            self.show_warning_dialog(
                f"NRP {nrp} tidak ditemukan dalam database.\nSilahkan tambahkan data karyawan terlebih dahulu."
            )
            return

        self.start_capture_process()

    def start_capture_process(self):
        self.is_registering_face = True
        self.face_encodings_list = []
        self.update_progress_bar(0)
        self.nrp_input.disabled = True
        self.register_button.disabled = True
        self.cancel_button.disabled = False
        self.instruction_main.text = "Sedang merekam wajah... Tetaplah menghadap kamera"

    def cancel_registration(self, *args):
        self.stop_face_registration(is_cancelled=True)

    def stop_face_registration(self, is_cancelled=False):
        self.is_registering_face = False
        self.nrp_input.disabled = False
        self.register_button.disabled = False
        self.cancel_button.disabled = True
        
        if is_cancelled:
            self.update_progress_bar(0)
            self.nrp_input.text = ""
            self.instruction_main.text = "Fokuskan Wajah Anda Pada Layar untuk Registrasi"
        else:
            self.update_progress_bar(100)
            self.show_registration_success_dialog()
            self.instruction_main.text = "Fokuskan Wajah Anda Pada Layar untuk Registrasi"

    def show_registration_success_dialog(self):
        content = MDBoxLayout(
            orientation="vertical",
            spacing=dp(20),
            padding=dp(30),
            size_hint_y=None,
            height=dp(300)
        )
        content.bind(minimum_height=content.setter('height'))

        icon = Image(
            source="assets/icon_checklist.png",
            size_hint=(None, None),
            size=(dp(100), dp(100)),
            pos_hint={"center_x": 0.5}
        )

        label = MDLabel(
            text="Registrasi Wajah Berhasil",
            halign="center",
            font_style="H5",
            theme_text_color="Primary",
            font_name="Poppins-Bold" if "Poppins-Bold" in LabelBase._fonts else "Roboto",
            size_hint_y=None,
            height=dp(40)
        )

        content.add_widget(icon)
        content.add_widget(label)

        dialog = MDDialog(
            type="custom",
            content_cls=content,
            size_hint=(0.8, None),
            height=dp(300),
            auto_dismiss=True
        )
        dialog.open()
        Clock.schedule_once(lambda dt: dialog.dismiss(), 2.0)

    def show_warning_dialog(self, message, confirm_action=None):
        buttons = [
            MDFlatButton(
                text="OK",
                on_release=lambda x: dialog.dismiss()
            )
        ]
        
        if confirm_action:
            buttons = [
                MDFlatButton(
                    text="Tidak",
                    on_release=lambda x: dialog.dismiss()
                ),
                MDRaisedButton(
                    text="Ya",
                    md_bg_color=(0.631, 0.694, 0.909, 1),
                    on_release=lambda x: [dialog.dismiss(), confirm_action()]
                )
            ]
        
        dialog = MDDialog(
            title="Peringatan",
            text=message,
            buttons=buttons
        )
        dialog.open()

    def show_keyboard(self, instance, value):
        if value:
            if self.vkeyboard is None:
                self.vkeyboard = VKeyboard()
                self.vkeyboard.size_hint = (1, 0.3)
                self.vkeyboard.bind(on_textinput=self.on_textinput)
                self.vkeyboard.bind(on_key_down=self.on_key_down)
                self.add_widget(self.vkeyboard)
        else:
            if self.vkeyboard:
                self.remove_widget(self.vkeyboard)
                self.vkeyboard = None

    def on_textinput(self, keyboard, text):
        if self.nrp_input.focus:
            self.nrp_input.text += text

    def on_key_down(self, keyboard, keycode, text, modifiers):
        if self.nrp_input.focus:
            if keycode == "backspace":
                self.nrp_input.text = self.nrp_input.text[:-1]
            elif keycode == "enter":
                self.nrp_input.focus = False
            elif keycode == "escape":
                self.nrp_input.focus = False

    def save_face_data(self):
        nrp = self.nrp_input.text.strip()

        if not nrp or not self.face_encodings_list:
            return

        mean_encoding = np.mean(self.face_encodings_list, axis=0).tolist()

        # Kirim ke API
        response = send_registration_to_api(nrp, mean_encoding)

        if response and response.get('status') == 200:
            # Reload data wajah
            thread = threading.Thread(target=load_face_data_from_api)
            thread.start()

            self.stop_face_registration(is_cancelled=False)
        else:
            error_msg = response.get('message', 'Gagal menyimpan data ke server') if response else 'Gagal menyimpan data ke server'
            self.show_result_dialog(
                error_msg,
                "assets/fail_icon.png",
                (0.921, 0.364, 0.380, 1)
            )

    def show_result_dialog(self, message, icon, color):
        content = MDBoxLayout(
            orientation="vertical",
            spacing=dp(20),
            padding=dp(20)
        )

        icon_widget = Image(
            source=icon,
            size_hint=(1, None),
            height=dp(100)
        )

        message_label = MDLabel(
            text=message,
            halign="center",
            theme_text_color="Custom",
            text_color=color,
            font_name="Poppins-Medium" if "Poppins-Medium" in LabelBase._fonts else "Roboto"
        )

        content.add_widget(icon_widget)
        content.add_widget(message_label)

        dialog = MDDialog(
            type="custom",
            content_cls=content,
            buttons=[
                MDRaisedButton(
                    text="Tutup",
                    md_bg_color=(0.631, 0.694, 0.909, 1),
                    on_release=lambda x: dialog.dismiss()
                )
            ]
        )
        dialog.open()

# ==================== SCREENS ====================
class MainScreen(Screen):
    def __init__(self, screen_manager, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation="horizontal")
        
        self.sidebar = Sidebar(screen_manager=screen_manager)
        self.main_content = MainContent(screen_manager=screen_manager)

        layout.add_widget(self.sidebar)
        layout.add_widget(self.main_content)

        self.add_widget(layout)

    def on_enter(self):
        self.main_content.on_enter()

    def on_leave(self):
        self.main_content.on_leave()

class RegisterScreen(Screen):
    def __init__(self, screen_manager, **kwargs):
        super().__init__(**kwargs)
        self.screen_manager = screen_manager
        self.sidebar = Sidebar(screen_manager=screen_manager)
        self.main_content = Registration(screen_manager=screen_manager)

        layout = BoxLayout(orientation="horizontal")
        layout.add_widget(self.sidebar)
        layout.add_widget(self.main_content)
        self.add_widget(layout)

    def on_enter(self):
        self.main_content.on_enter()

    def on_leave(self):
        self.main_content.on_leave()

# ==================== MAIN APP ====================
class myapp(MDApp):
    def build(self):
        sm = ScreenManager()

        main_screen = MainScreen(name="main", screen_manager=sm)
        sm.add_widget(main_screen)

        register_screen = RegisterScreen(name="register", screen_manager=sm)
        sm.add_widget(register_screen)

        sm.current = "main"
        
        # Load data wajah dari API saat startup
        threading.Thread(target=load_face_data_from_api).start()
        
        return sm

if __name__ == '__main__':
    myapp().run()