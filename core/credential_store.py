"""
API Key 安全存储 — Credential Store

设计目标：
1. API Key 永远不写入日志/屏幕截图
2. 支持多层级存储（系统钥匙串 → 加密文件 → 明文 fallback）
3. UI 层只需调用 save / load / delete，不需要关心底层
"""

import os
import json
import base64
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


# ─── 存储后端 ─────────────────────────────────────────────

class CredentialBackend:
    """凭证存储后端基类"""

    def save(self, key: str, value: str) -> bool:
        raise NotImplementedError

    def load(self, key: str) -> Optional[str]:
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


class KeychainBackend(CredentialBackend):
    """
    系统钥匙串存储（最安全）
    使用 keyring 库，自动适配：
      - macOS: Keychain
      - Windows: Credential Manager
      - Linux: Secret Service (GNOME Keyring / KDE Wallet)
    """

    SERVICE_NAME = "negative-color-corrector"

    def __init__(self):
        self._available = None

    def _check_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import keyring
            # 尝试访问钥匙串
            keyring.get_password(self.SERVICE_NAME, "test")
            self._available = True
        except Exception:
            self._available = False
        return self._available

    def save(self, key: str, value: str) -> bool:
        if not self._check_available():
            return False
        try:
            import keyring
            keyring.set_password(self.SERVICE_NAME, key, value)
            return True
        except Exception:
            return False

    def load(self, key: str) -> Optional[str]:
        if not self._check_available():
            return None
        try:
            import keyring
            return keyring.get_password(self.SERVICE_NAME, key)
        except Exception:
            return None

    def delete(self, key: str) -> bool:
        if not self._check_available():
            return False
        try:
            import keyring
            keyring.delete_password(self.SERVICE_NAME, key)
            return True
        except Exception:
            return False

    @property
    def name(self) -> str:
        return "系统钥匙串"


class EncryptedFileBackend(CredentialBackend):
    """
    加密文件存储（备选方案）
    使用 Fernet (对称加密) 加密后写入本地文件。
    解密密钥存储为环境变量或派生自机器特征。
    """

    def __init__(self, file_path: str = ""):
        self.file_path = file_path or os.path.join(
            os.path.dirname(__file__), "..", ".credentials.enc"
        )

    def _get_fernet(self):
        """获取加密器

        密钥来源（按优先级）：
        1. 环境变量 NCC_CRED_KEY（用户自定义）
        2. 自动生成（基于机器 ID + 固定 salt）
        """
        try:
            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        except ImportError:
            raise RuntimeError(
                "需要安装 cryptography: pip install cryptography"
            )

        # 获取密钥材料
        key_material = os.environ.get("NCC_CRED_KEY")
        if not key_material:
            # 使用机器特征组合（非完美安全，但防止随意的明文泄露）
            machine_id = self._get_machine_id()
            key_material = f"ncc-v1-{machine_id}"

        # 派生 32 字节密钥
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"ncc-cred-salt-v1",
            iterations=100_000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(key_material.encode()))

        # 缓存 Fernet 实例
        if not hasattr(self, "_fernet_cache"):
            self._fernet_cache = {}
        cache_key = key_material[:16]
        if cache_key not in self._fernet_cache:
            self._fernet_cache[cache_key] = Fernet(key)
        return self._fernet_cache[cache_key]

    def _get_machine_id(self) -> str:
        """获取机器唯一标识"""
        # macOS
        if os.path.exists("/etc/machine-id"):
            return open("/etc/machine-id").read().strip()
        # Linux
        if os.path.exists("/var/lib/dbus/machine-id"):
            return open("/var/lib/dbus/machine-id").read().strip()
        # macOS fallback
        import subprocess
        try:
            result = subprocess.run(
                ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                capture_output=True, text=True
            )
            for line in result.stdout.split("\n"):
                if "IOPlatformUUID" in line:
                    return line.split('"')[3]
        except Exception:
            pass
        # 最终 fallback：使用 hostname
        import socket
        return socket.gethostname()

    def _load_all(self) -> dict:
        """加载整个加密文件"""
        path = Path(self.file_path)
        if not path.exists():
            return {}
        try:
            fernet = self._get_fernet()
            encrypted_data = path.read_bytes()
            decrypted = fernet.decrypt(encrypted_data)
            return json.loads(decrypted.decode())
        except Exception:
            # 解密失败（密钥变了？），返回空
            return {}

    def _save_all(self, data: dict) -> bool:
        """保存整个加密文件"""
        try:
            fernet = self._get_fernet()
            path = Path(self.file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            json_data = json.dumps(data).encode()
            encrypted = fernet.encrypt(json_data)
            path.write_bytes(encrypted)
            # 设置文件权限（仅用户可读写）
            path.chmod(0o600)
            return True
        except Exception as e:
            print(f"[CredentialStore] 加密保存失败: {e}")
            return False

    def save(self, key: str, value: str) -> bool:
        data = self._load_all()
        # 用 provider 名作为 key 前缀，方便管理
        data[key] = value
        return self._save_all(data)

    def load(self, key: str) -> Optional[str]:
        data = self._load_all()
        return data.get(key)

    def delete(self, key: str) -> bool:
        data = self._load_all()
        if key in data:
            del data[key]
            return self._save_all(data)
        return True

    @property
    def name(self) -> str:
        return "加密文件"


class PlaintextFileBackend(CredentialBackend):
    """
    JSON 明文文件（最后的 fallback / 开发者调试用）
    会弹出警告提示用户这不安全。
    """

    def __init__(self, file_path: str = ""):
        self.file_path = file_path or os.path.join(
            os.path.dirname(__file__), "..", ".credentials.json"
        )
        self._warned = False

    def _warn(self):
        if not self._warned:
            print(
                "\n⚠️  警告：API Key 以明文存储在本地文件！\n"
                "   建议安装 keyring 库或 cryptography 库来加密存储:\n"
                "   pip install keyring\n"
                "   或\n"
                "   pip install cryptography\n"
            )
            self._warned = True

    def _load_all(self) -> dict:
        path = Path(self.file_path)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, PermissionError):
            return {}

    def _save_all(self, data: dict) -> bool:
        try:
            path = Path(self.file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
            path.chmod(0o600)
            return True
        except Exception:
            return False

    def save(self, key: str, value: str) -> bool:
        self._warn()
        data = self._load_all()
        data[key] = value
        return self._save_all(data)

    def load(self, key: str) -> Optional[str]:
        self._warn()
        data = self._load_all()
        return data.get(key)

    def delete(self, key: str) -> bool:
        data = self._load_all()
        if key in data:
            del data[key]
            return self._save_all(data)
        return True

    @property
    def name(self) -> str:
        return "明文文件 (⚠️ 不安全)"


# ─── 统一存储入口 ─────────────────────────────────────────

class CredentialStore:
    """
    API Key 统一存储入口

    自动选择最佳可用后端：
      系统钥匙串 → 加密文件 → 明文文件
    UI 只需要调用 store.save() / store.load() / store.delete()
    """

    def __init__(self, cred_dir: str = ""):
        self.cred_dir = cred_dir or os.path.join(
            os.path.dirname(__file__), ".."
        )
        self._backend: Optional[CredentialBackend] = None

    @property
    def backend(self) -> CredentialBackend:
        if self._backend is None:
            self._backend = self._select_backend()
        return self._backend

    def _select_backend(self) -> CredentialBackend:
        """按优先级选择最佳可用后端"""
        # 1st: 系统钥匙串
        kb = KeychainBackend()
        if kb._check_available():
            return kb

        # 2nd: 加密文件 (需要 cryptography)
        ef = EncryptedFileBackend(
            file_path=os.path.join(self.cred_dir, ".credentials.enc")
        )
        try:
            # 测试能否正常使用
            ef.save("_test_", "_test_")
            ef.delete("_test_")
            return ef
        except Exception:
            pass

        # 3rd: 明文文件（最后的 fallback）
        pf = PlaintextFileBackend(
            file_path=os.path.join(self.cred_dir, ".credentials.json")
        )
        return pf

    @property
    def backend_name(self) -> str:
        """返回当前使用的后端名称（供 UI 显示）"""
        return self.backend.name

    def save(self, provider_id: str, api_key: str) -> bool:
        """保存 API Key

        Args:
            provider_id: Provider 标识 (如 "openrouter")
            api_key: API Key 字符串

        Returns:
            是否保存成功
        """
        key_name = f"api_key_{provider_id}"
        return self.backend.save(key_name, api_key)

    def load(self, provider_id: str) -> Optional[str]:
        """加载 API Key

        Args:
            provider_id: Provider 标识

        Returns:
            API Key，如果不存在返回 None
        """
        key_name = f"api_key_{provider_id}"
        return self.backend.load(key_name)

    def delete(self, provider_id: str) -> bool:
        """删除 API Key"""
        key_name = f"api_key_{provider_id}"
        return self.backend.delete(key_name)

    def has_key(self, provider_id: str) -> bool:
        """检查是否已保存 API Key"""
        return self.load(provider_id) is not None
