from typing import Any
import torch
import random
import numpy as np
import numbers
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import torchvision.transforms as torch_tr
import os
from torch.backends import cudnn

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
# cudnn.deterministic = True
# cudnn.benchmark = False
try:
    import accimage
except ImportError:
    accimage = None

class HPF(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        fft_img = np.fft.fftn(img)
        fft_shift = np.fft.fftshift(fft_img)
        #print(type(img))
        cols, rows = img.size
        crow, ccol = int(rows / 2), int(cols / 2)

        mask_HPF = np.ones((rows, cols, 3), np.uint8)
        r = 16
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area_HPF = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
        mask_HPF[mask_area_HPF] = 0
        fshift_HPF = fft_shift * mask_HPF
        ifft_shift = np.fft.ifftshift(fshift_HPF)
        img = np.fft.ifftn(ifft_shift)
        img = np.array(img).astype(np.float32)
        
        return {'image': img, 'label': mask}
    
class PHOT(object):
    def __call__(self,sample):
        img = sample['image']
        mask = sample['label']
        fft_img = np.fft.fftn(img)
        fft_amp = np.abs(fft_img)
        phase_img = np.fft.ifftn((fft_img/fft_amp))
        phase_img = (phase_img*5*255).astype(np.float32)

        return {'image': phase_img, 'label': mask}

    
class LPF(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        fft_img = np.fft.fftn(img)
        fft_shift = np.fft.fftshift(fft_img)
        cols, rows = img.size
        crow, ccol = int(rows / 2), int(cols / 2)

        mask_LPF = np.ones((rows, cols, 3), np.uint8)
        r = 16
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area_HPF = (x - center[0]) ** 2 + (y - center[1]) ** 2 >= r*r
        mask_LPF[mask_area_HPF] = 0
        fshift_LPF = fft_shift * mask_LPF
        ifft_shift = np.fft.ifftshift(fshift_LPF)
        img = np.fft.ifftn(ifft_shift)
        img = np.array(img).astype(np.float32)

        return {'image': img, 'label': mask}



class Contrast(object):

    def __call__(self,sample):
        img = sample['image']
        mask = sample['label']
        #print(type(img))
        #img = Image.fromarray(img.astype('uint8'), 'RGB')
        img = ImageEnhance.Contrast(img).enhance(2.0)
        img = np.array(img)
        return {'image': img,
                 'label': mask}

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #print('sample',sample)
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        # print('sample',sample)
        img = sample['image']
        mask = sample['label']
        # print(type(img))
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': mask}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}
    
class RandomCrop_p(object):
    def __init__(self, base_size, crop_size):
        self.base_size = base_size
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size

        x0 = random.randint(0, w - self.crop_size)
        y0 = random.randint(0, h - self.base_size)
        
        mask = mask.crop((x0, y0, x0 + self.crop_size, y0 + self.base_size))
        
        img = img.crop((x0, y0, x0 + self.crop_size, y0 + self.base_size))

        return {'image': img,
                'label': mask}
    
class RandomCrop_p2(object):
    def __init__(self, crop_sizew, crop_sizeh):
        self.crop_sizew = crop_sizew
        self.crop_sizeh = crop_sizeh
        

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size

        x0 = random.randint(0, w - self.crop_sizew)
        y0 = random.randint(0, h - self.crop_sizeh)
        
        mask = mask.crop((x0, y0, x0 + self.crop_sizew, y0 + self.crop_sizeh))
        
        img = img.crop((x0, y0, x0 + self.crop_sizew, y0 + self.crop_sizeh))

        return {'image': img,
                'label': mask}
    
class RandomCrop(object):
    """
    Take a random crop from the image.

    First the image or crop size may need to be adjusted if the incoming image
    is too small...

    If the image is smaller than the crop, then:
         the image is padded up to the size of the crop
         unless 'nopad', in which case the crop size is shrunk to fit the image

    A random crop is taken such that the crop fits within the image.
    If a centroid is passed in, the crop must intersect the centroid.
    """
    def __init__(self, size, ignore_index=0, nopad=True):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.ignore_index = ignore_index
        self.nopad = nopad
        self.pad_color = (0, 0, 0)

    def __call__(self, img, mask, centroid=None, pos=None):
        assert img.size == mask.size
        w, h = img.size
        # ASSUME H, W
        th, tw = self.size
        if w == tw and h == th:
            if pos is not None:
                return img, mask, pos
            return img, mask

        if self.nopad:
            if th > h or tw > w:
                # Instead of padding, adjust crop size to the shorter edge of image.
                shorter_side = min(w, h)
                th, tw = shorter_side, shorter_side
        else:
            # Check if we need to pad img to fit for crop_size.
            if th > h:
                pad_h = (th - h) // 2 + 1
            else:
                pad_h = 0
            if tw > w:
                pad_w = (tw - w) // 2 + 1
            else:
                pad_w = 0
            border = (pad_w, pad_h, pad_w, pad_h)
            if pad_h or pad_w:
                img = ImageOps.expand(img, border=border, fill=self.pad_color)
                mask = ImageOps.expand(mask, border=border, fill=self.ignore_index)
                w, h = img.size
                if pos is not None:
                    pos = ImageOps.expand(pos[0], border=border, fill=IGNORE_POS), ImageOps.expand(pos[1], border=border, fill=IGNORE_POS)

        if centroid is not None:
            # Need to insure that centroid is covered by crop and that crop
            # sits fully within the image
            c_x, c_y = centroid
            max_x = w - tw
            max_y = h - th
            x1 = random.randint(c_x - tw, c_x)
            x1 = min(max_x, max(0, x1))
            y1 = random.randint(c_y - th, c_y)
            y1 = min(max_y, max(0, y1))
        else:
            if w == tw:
                x1 = 0
            else:
                x1 = random.randint(0, w - tw)
            if h == th:
                y1 = 0
            else:
                y1 = random.randint(0, h - th)

        if pos is not None:
            pos = pos[0].crop((x1, y1, x1 + tw, y1 + th)), pos[1].crop((x1, y1, x1 + tw, y1 + th))
            return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th)), pos

        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))
    
class ResizeHeight(object):
    def __init__(self, size):
        self.target_h = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        target_w = int(w / h * self.target_h)
        img = img.resize((target_w, self.target_h), Image.BICUBIC)
        mask= mask.resize((target_w, self.target_h), Image.NEAREST)

        return {'image': img,
                'label': mask}
    
class CenterCropPad(object):
    def __init__(self, size, ignore_index=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.ignore_index = ignore_index

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size
        w, h = img.size
        if isinstance(self.size, tuple):
                tw, th = self.size[0], self.size[1]
        else:
                th, tw = self.size, self.size
	

        if w < tw:
            pad_x = tw - w
        else:
            pad_x = 0
        if h < th:
            pad_y = th - h
        else:
            pad_y = 0

        if pad_x or pad_y:
            # left, top, right, bottom
            img = ImageOps.expand(img, border=(pad_x, pad_y, pad_x, pad_y), fill=0)
            mask = ImageOps.expand(mask, border=(pad_x, pad_y, pad_x, pad_y),
                                   fill=self.ignore_index)

        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        mask= mask.crop((x1, y1, x1 + tw, y1 + th))

        return {'image': img,
                'label': mask}



class RandomSizeAndCrop(object):
    def __init__(self, size, crop_nopad,
                 scale_min=0.5, scale_max=2.0, ignore_index=0, pre_size=None):
        self.size = size
        self.crop = RandomCrop(self.size, ignore_index=ignore_index, nopad=crop_nopad)
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.pre_size = pre_size

    def __call__(self, sample, centroid=None, pos=None):
        img = sample['image']
        mask = sample['label']
        try:
            assert img.size == mask.size
        except:
            pass
        # first, resize such that shorter edge is pre_size
        if self.pre_size is None:
            scale_amt = 1.
        elif img.size[1] < img.size[0]:
            scale_amt = self.pre_size / img.size[1]
        else:
            scale_amt = self.pre_size / img.size[0]
        scale_amt *= random.uniform(self.scale_min, self.scale_max)
        w, h = [int(i * scale_amt) for i in img.size]

        if centroid is not None:
            centroid = [int(c * scale_amt) for c in centroid]

        img, mask = img.resize((w, h), Image.BICUBIC), mask.resize((w, h), Image.NEAREST)

        if pos is not None:
            pos = pos[0].resize((w, h), Image.NEAREST), pos[1].resize((w, h), Image.NEAREST)

        img,mask =  self.crop(img, mask, centroid, pos)
        return {'image': img,
                'label': mask}




class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        #print("w and h",w, h)
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
            #print("ow: ", ow, w, oh, h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}

class Resize(object):
    def __init__(self, size1,size2):
        self.size = (size1, size2)  # size: (h, w)
        
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        try:
            assert img.size == mask.size
        except:
            pass

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)
        
        return {'image': img,
                'label': mask}
    
def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)
    
def adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.
    Args:
        img (PIL Image): PIL Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.
    Returns:
        PIL Image: Brightness adjusted image.
    """
    image = img['image']
    mask = img['label']
    if not _is_pil_image(image):
        raise TypeError('img should be PIL Image. Got {}'.format(type(image)))

    enhancer = ImageEnhance.Brightness(image)
    img = enhancer.enhance(brightness_factor)
    
    return {'image': img,
            'label': mask}


def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an Image.
    Args:
        img (PIL Image): PIL Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
    Returns:
        PIL Image: Contrast adjusted image.
    """
    image = img['image']
    mask = img['label']
    if not _is_pil_image(image):
        raise TypeError('img should be PIL Image. Got {}'.format(type(image)))

    enhancer = ImageEnhance.Contrast(image)
    img = enhancer.enhance(contrast_factor)
    return {'image': img,
            'label': mask}


def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.
    Args:
        img (PIL Image): PIL Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.
    Returns:
        PIL Image: Saturation adjusted image.
    """
    image = img['image']
    mask = img['label']
    if not _is_pil_image(image):
        raise TypeError('img should be PIL Image. Got {}'.format(type(image)))

    enhancer = ImageEnhance.Color(image)
    img = enhancer.enhance(saturation_factor)
    return {'image': img,
            'label': mask}


def adjust_hue(img, hue_factor):
    """Adjust hue of an image.
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.
    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.
    See https://en.wikipedia.org/wiki/Hue for more details on Hue.
    Args:
        img (PIL Image): PIL Image to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.
    Returns:
        PIL Image: Hue adjusted image.
    """
    image = img['image']
    mask = img['label']
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    if not _is_pil_image(image):
        raise TypeError('img should be PIL Image. Got {}'.format(type(image)))
    input_mode = image.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return image

    h, s, v = image.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')
    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return {'image': img,
            'label': mask}


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        if brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(
                torch_tr.Lambda(lambda img: adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(
                torch_tr.Lambda(lambda img: adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(
                torch_tr.Lambda(lambda img: adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = np.random.uniform(-hue, hue)
            transforms.append(
                torch_tr.Lambda(lambda img: adjust_hue(img, hue_factor)))

        np.random.shuffle(transforms)
        transform = torch_tr.Compose(transforms)

        return transform

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Input image.
        Returns:
            PIL Image: Color jittered image.
        """
        if random.random() < 0.5:
            transform = self.get_params(self.brightness, self.contrast,
                                        self.saturation, self.hue)
            return transform(img)
        else:
            return img
