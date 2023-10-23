import cv2
import numpy as np
import matplotlib.pyplot as plt


def lowhigh_filter(cutoff_frequency, filtering):

    if (filtering == 'lowpass'):
        mask = np.sqrt(U**2 + V**2) <= cutoff_frequency
    else:
        mask = np . sqrt ( U **2 + V **2) > cutoff_frequency
        

    dft_shift_filtered = dft_shift * mask[:, :, np.newaxis]
    idft = np.fft.ifftshift(dft_shift_filtered)
    
    img_filtered = cv2.idft(idft)
    img_filtered = cv2.magnitude(img_filtered[:, :, 0], img_filtered[:, :, 1])

    return img_filtered, dft_shift_filtered, mask

def band_filter(mask):
    dft_shift_filtered = dft_shift * mask[:, :, np.newaxis]
    idft = np.fft.ifftshift(dft_shift_filtered)
    
    img_filtered = cv2.idft(idft)
    img_filtered = cv2.magnitude(img_filtered[:, :, 0], img_filtered[:, :, 1])

    return img_filtered, dft_shift_filtered

def notch_filter(notch_freq):
    # mask
    cM, cN = M//2, N//2 # menghitung titik tengah dari gambar

    mask = np.ones((M, N, 2), np.uint8)
    center = [cM, cN]
    x, y = np.ogrid[:M, :N]
    mask_con = (np.sqrt((x - center[0])**2 + (y - center[1])**2) <= notch_freq) & (np.sqrt((x - center[0])**2 + (y - center[1])**2) >= notch_freq)
    mask[mask_con] = 0

    dft_shift_filtered = dft_shift * mask
    idft = np.fft.ifftshift(dft_shift_filtered)
    
    img_filtered = cv2.idft(idft)
    img_filtered = cv2.magnitude(img_filtered[:,:,0], img_filtered[:,:,1])

    spectrum = 20*np.log(cv2.magnitude(dft_shift_filtered[:,:,0], dft_shift_filtered[:,:,1]))

    return img_filtered, dft_shift_filtered, mask, spectrum


def plot_spectrum(dft_shift):
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1] + 1e-5))  # Menambahkan konstanta kecil untuk menghindari log(0)
    return magnitude_spectrum


defimg = cv2.imread('Tower2ITS.jpg', cv2.IMREAD_GRAYSCALE)
lp_cutoff = 20
hp_cutoff = 200
notch_cutoff = 70

#DFT
dft = cv2.dft(np.float32(defimg), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
default_spektrum = plot_spectrum(dft_shift)

# bikin mask
M, N = defimg.shape
u = np.arange(M)
v = np.arange(N)
u = u - M//2
v = v - N//2
U, V = np.meshgrid(v, u)

##Lowpass & Highpass################
lowpass_filtered, dft_lowpass, mask_lowpass = lowhigh_filter(lp_cutoff, 'lowpass')
lowpass_spektrum = plot_spectrum(dft_lowpass)

highpass_filtered, dft_highpass, mask_highpass = lowhigh_filter(hp_cutoff, 'highpass')
highpass_spektrum = plot_spectrum(dft_highpass)

##Bandreject & Bandpass#############
mask_reject = mask_lowpass + mask_highpass
bandreject_filtered, dft_bandreject = band_filter(mask_reject)
bandreject_spektrum = plot_spectrum(dft_bandreject)

mask_pass = 1 - mask_reject
bandpass_filtered, dft_bandpass = band_filter(mask_pass)
bandpass_spektrum = plot_spectrum(dft_bandpass)

##Notch filter######################
notch_filtered, dft_notch, mask_notch, notch_spektrum = notch_filter(notch_cutoff)

# First Tab
plt.figure(figsize=(10, 4))
plt.suptitle('Tab 1: Default Image')

plt.subplot(1, 2, 1)
plt.imshow(defimg, cmap='gray')
plt.title('Image')
plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2)
plt.imshow(default_spektrum, cmap='gray')
plt.title('Spektrum')
plt.xticks([]), plt.yticks([])

# Second Tab
plt.figure(figsize=(10, 6))
plt.suptitle('Tab 2: Mask Filter')

plt.subplot(2, 2, 1)
plt.imshow(mask_lowpass, cmap='gray')
plt.title('Mask lowpass')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2)
plt.imshow(mask_highpass, cmap='gray')
plt.title('Mask Highpass')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 3)
plt.imshow(mask_reject, cmap='gray')
plt.title('Masker Bandreject')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 4)
plt.imshow(mask_pass, cmap='gray')
plt.title('Mask Bandpass')
plt.xticks([]), plt.yticks([])

# Third Tab
plt.figure(figsize=(10, 6))
plt.suptitle('Tab 3: Bandreject & Bandpass Filter')

plt.subplot(2, 2, 1)
plt.imshow(bandreject_filtered, cmap='gray')
plt.title('Hasil Bandreject')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2)
plt.imshow(bandpass_filtered, cmap='gray')
plt.title('Hasil Bandpass')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 3)
plt.imshow(bandreject_spektrum, cmap='gray')
plt.title('Spektrum Bandreject')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 4)
plt.imshow(bandpass_spektrum, cmap='gray')
plt.title('Spektrum Bandpass')
plt.xticks([]), plt.yticks([])

# Fourth Tab
plt.figure(figsize=(10, 6))
plt.suptitle('Tab 4: Lowpass & Highpass Filter')

plt.subplot(2, 2, 1)
plt.imshow(lowpass_filtered, cmap='gray')
plt.title('Hasil Lowpass')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2)
plt.imshow(highpass_filtered, cmap='gray')
plt.title('Hasil Highpass')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 3)
plt.imshow(lowpass_spektrum, cmap='gray')
plt.title('Spektrum Lowpass')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 4)
plt.imshow(highpass_spektrum, cmap='gray')
plt.title('Spektrum Highpass')
plt.xticks([]), plt.yticks([])

# Fifth Tab
plt.figure(figsize=(10, 6))
plt.suptitle('Tab 5: Notch filter')

plt.subplot(2, 2, 1)
plt.imshow(notch_filtered, cmap='gray')
plt.title('Hasil notchpass')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2)
plt.imshow(notch_spektrum, cmap='gray')
plt.title('Spektrum notch')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 3)
plt.imshow(mask_notch[:,:,0], cmap='gray')
plt.title('Mask Notch')
plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()