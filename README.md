Innanzitutto installare dalle API di Google il kernel Linux Jammy Desktop 5.10.160.

Di seguito sono presentate tutte le librerie utilizzate per il progetto.

sudo apt update
sudo apt install pip
 
pip install numpy
pip install tensorflow
pip install tensorflow_hub
sudo apt install stress-ng
sudo apt install rt-tests
 
wget https://github.com/rockchip-linux/rknn-toolkit2/archive/refs/heads/master.zip
 
unzip master.zip
 
mkdir -p ~/rknn_runtime_rk3588 && cd ~/rknn_runtime_rk3588
 
git clone https://github.com/rockchip-linux/rknpu2.git
 
sudo cp /home/orangepi/rknn_runtime_rk3588/rknpu2/runtime/RK3588/Linux/librknn_api/aarch64/librknnrt.so /usr/lib/
 
pip3 install rknn-toolkit-lite2
 
pip install opencv-python
 
*PER APPLICARE LA PATCH REAL TIME:*
 
cd
cd *cartella Kernel Real Time -> rt-with-updated-drivers*
 
sudo dpkg -i linux-image-legacy-rockchip-rk3588_23.05.0-trunk--5.10.110-Scbae-De8b2-Pb119-C67d9Hfe66-Vc222-Be9aa_arm64.deb linux-dtb-legacy-rockchip-rk3588_23.05.0-trunk--5.10.110-Scbae-De8b2-Pb119-C67d9Hfe66-Vc222-Be9aa_arm64.deb linux-headers-legacy-rockchip-rk3588_23.05.0-trunk--5.10.110-Scbae-De8b2-Pb119-C67d9Hfe66-Vc222-Be9aa_arm64.deb
(E' UN UNICO COMANDO)
 
sudo reboot
