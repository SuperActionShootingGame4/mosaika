"""PyQt6 PyInstaller collection diagnostic."""
from PyInstaller.utils.hooks import collect_all

d, b, h = collect_all('PyQt6')
print(f"collect_all => datas:{len(d)} binaries:{len(b)} hiddenimports:{len(h)}")
print("first 3 datas:", d[:3])
print("first 3 binaries:", b[:3])
print("first 3 hiddenimports:", h[:3])
