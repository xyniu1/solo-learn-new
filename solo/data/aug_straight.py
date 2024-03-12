import math
import torch
from torch import Tensor
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F
from typing import List, Tuple
from PIL import ImageOps, ImageFilter


class Translation():
    def __init__(
        self,
        size,
        scale_low, # scale of the crop window (to be translated)
        scale_high,
        intensity, # min intensity of the displacement of the window, intensity * (width or height)
        t, # length = t + 1
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
        ):
        super().__init__()
        self.size = size
        self.scale = [scale_low, scale_high]
        self.intensity = intensity
        self.t = t
        self.ratio = ratio
        self.interpolation = interpolation
        self.antialias = antialias

    @staticmethod
    def get_params(img: Tensor, scale: List[float], ratio: List[float], intensity:float, t:float) -> Tuple[int, int, int, int]:
        _, height, width = F.get_dimensions(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for it in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            
            if not (0 < w <= width and 0 < h <= height) and it < 9:
                continue
            
            # for the last iteration
            if not (0 < w <= width and 0 < h <= height) and it == 9:
                in_ratio = float(width) / float(height)
                if in_ratio < min(ratio):
                    w = width
                    h = int(round(w / min(ratio)))
                elif in_ratio > max(ratio):
                    h = height
                    w = int(round(h * max(ratio)))
                else:  # whole image
                    w = width
                    h = height
                    
            # in the strong version, decide dx and dy first, then decide x and y
            dx_max = math.floor((width-w) / t)
            dy_max = math.floor((height-h) / t)
            
            # if one of (dx, dy) is above threshold, then the other one can be any number
            if all([dx_max < intensity * width, dy_max < intensity * height, it < 9]):
                continue
            elif dx_max >= intensity * width and dy_max < intensity * height:
                dx_min = math.ceil(intensity * width)
                dy_min = 0
            elif dx_max < intensity * width and dy_max >= intensity * height:
                dy_min = math.ceil(intensity * height)
                dx_min = 0
            elif dx_max >= intensity * width and dy_max >= intensity * height:
                if torch.rand(1).item() < 0.5:
                    dx_min = math.ceil(intensity * width)
                    dy_min = 0
                else:
                    dy_min = math.ceil(intensity * height)
                    dx_min = 0
            else: # for the last iteration
                dx_min = dx_max
                dy_min = dy_max
            
            dx = torch.randint(dx_min, dx_max + 1, size=(1,)).item()
            dy = torch.randint(dy_min, dy_max + 1, size=(1,)).item()
            
            # dx = torch.randint(0, dx_max + 1, size=(1,)).item()
            # dy = torch.randint(0, dy_max + 1, size=(1,)).item()
            
            if torch.rand(1).item() < 0.5:
                dx *= -1
            if torch.rand(1).item() < 0.5:
                dy *= -1
            
                
            x_min = max(-t*dx, 0)
            x_max = min(width-w-t*dx, width-w)
            y_min = max(-t*dy, 0)
            y_max = min(height-h-t*dy, height-h)
            x = torch.randint(x_min, x_max + 1, size=(1,)).item()
            y = torch.randint(y_min, y_max + 1, size=(1,)).item()
            
            # make sure the augmentation is strong enough
            if abs(dx) > intensity * width or abs(dy) > intensity * height:
                return h, w, x, y, dx, dy
            elif it == 9:
                return h, w, x, y, dx, dy


    def __call__(self, img):
        """
        Returns:
            A list of PIL Image or Tensor: Randomly cropped and resized image.
        """
        h, w, x, y, dx, dy = self.get_params(img, self.scale, self.ratio, self.intensity, self.t)
        img_seq = []
        for i in range(self.t + 1):
            img_seq.append(F.resized_crop(img, y+i*dy, x+i*dx, h, w, 
                                          [self.size, self.size], self.interpolation, antialias=self.antialias))
        return img_seq
    
    
class Tofro():
    def __init__(
        self,
        size,
        scale_low, # scale of the crop window (to be translated)
        scale_high,
        intensity, # min intensity of the displacement of the window, intensity * (width or height)
        t, # length = t + 1
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
        ):
        super().__init__()
        self.size = size
        self.scale = [scale_low, scale_high]
        self.intensity = intensity
        self.t = t
        self.ratio = ratio
        self.interpolation = interpolation
        self.antialias = antialias

    @staticmethod
    def get_params(img: Tensor, scale: List[float], ratio: List[float], intensity:float, t:float) -> Tuple[int, int, int, int]:

        _, height, width = F.get_dimensions(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for it in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            aspect_ratio = w / h
            
            if not (0 < w <= width and 0 < h <= height) and it < 9:
                continue
            
            # for the last iteration
            if not (0 < w <= width and 0 < h <= height) and it == 9:
                in_ratio = float(width) / float(height)
                if in_ratio < min(ratio):
                    w = width
                    h = int(round(w / min(ratio)))
                elif in_ratio > max(ratio):
                    h = height
                    w = int(round(h * max(ratio)))
                else:  # whole image
                    w = width
                    h = height
                aspect_ratio = w / h
                    
            h_max = min(height, int(width / aspect_ratio))
                    
            dh_max = math.floor((h_max - h) / t)
            dw_max = math.floor(dh_max * aspect_ratio)
            
            if all([dh_max < intensity * height, dw_max < intensity * width, it < 9]):
                continue
            elif dh_max >= intensity * height and dw_max < intensity * width:
                dh_min = math.ceil(intensity * height)
            elif dh_max < intensity * height and dw_max >= intensity * width:
                dh_min = math.ceil(intensity * width / aspect_ratio)
            elif dh_max >= intensity * height and dw_max >= intensity * width:
                if torch.rand(1).item() < 0.5:
                    dh_min = math.ceil(intensity * height)
                else:
                    dh_min = math.ceil(intensity * width / aspect_ratio)
            else: # for the last iteration
                dh_min = dh_max
            if dh_min > dh_max:
                dh = dh_max
            else:
                dh = torch.randint(dh_min, dh_max + 1, size=(1,)).item()
            
            # dh = torch.randint(0, dh_max + 1, size=(1,)).item()
            dw = math.floor(dh * aspect_ratio)
            
            x_min = math.ceil(max(0, int(dw*t/2.0)))
            x_max = math.floor(min(width-w, width-w-dw*t/2.0))
            y_min = math.ceil(max(0, int(dh*t/2.0)))
            y_max = math.floor(min(height-h, height-h-dh*t/2.0))

            x = torch.randint(x_min, x_max + 1, size=(1,)).item()
            y = torch.randint(y_min, y_max + 1, size=(1,)).item()
                
            # make sure the augmentation is strong enough
            if abs(dh) > intensity * height or abs(dw) > intensity * width:
                return h, w, x, y, dh, dw
            elif it == 9:
                return h, w, x, y, dh, dw


    def __call__(self, img):
        """
        Returns:
            A list of PIL Image or Tensor: Randomly cropped and resized image.
        """
        h, w, x, y, dh, dw = self.get_params(img, self.scale, self.ratio, self.intensity, self.t)
        img_seq = []
        for i in range(self.t + 1):
            img_seq.append(F.resized_crop(img, y-int(dh/2.0*i), x-int(dw/2.0*i), h+dh*i, w+dw*i,
                                          [self.size, self.size], self.interpolation, antialias=self.antialias))
        if torch.rand(1).item() < 0.5:
            img_seq = img_seq[::-1]
        return img_seq
    
    
class Rotation():
    def __init__(
        self,
        size,
        scale_low, # scale of the crop window (to be translated)
        scale_high,
        intensity, # min intensity of the displacement of the window, intensity * (width or height)
        t, # length = t + 1
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
        ):
        super().__init__()
        self.size = size
        self.scale = (scale_low, scale_high)
        self.intensity = intensity
        self.t = t
        self.ratio = ratio
        self.interpolation = interpolation
        self.antialias = antialias

    def __call__(self, img):
        """
        Returns:
            A list of PIL Image or Tensor: Randomly cropped and resized image.
        """
        eps = 1e-6
        dtheta = torch.empty(1).uniform_(self.intensity * 180, 180 / self.t).item()
        theta = torch.empty(1).uniform_(-90, 90 - self.t * dtheta + eps).item()
        img = transforms.RandomResizedCrop(self.size, self.scale, self.ratio, self.interpolation, self.antialias)(img)
        img_seq = []
        for i in range(self.t + 1):
            angle = theta + i * dtheta
            angle = angle + 360 if angle < 0 else angle
            img_seq.append(F.rotate(img, angle))
        if torch.rand(1).item() < 0.5:
            img_seq = img_seq[::-1]
        return img_seq
    
    
class StraightTransform():
    def __init__(
        self,
        size,
        scale_low_translation,
        scale_high_translation,
        scale_low_tofro,
        scale_high_tofro,
        scale_low_rotation,
        scale_high_rotation,
        intensity_translation,
        intensity_tofro,
        intensity_rotation,
        t,        
        p=[1.0/3.0, 1.0/3.0] # prob of translation, tofro
        ):
        super().__init__()
        self.size = size
        self.scale_low_translation = scale_low_translation
        self.scale_high_translation = scale_high_translation
        self.scale_low_tofro = scale_low_tofro
        self.scale_high_tofro = scale_high_tofro
        self.scale_low_rotation = scale_low_rotation
        self.scale_high_rotation = scale_high_rotation
        self.intensity_translation = intensity_translation
        self.intensity_tofro = intensity_tofro
        self.intensity_rotation = intensity_rotation
        self.t = t
        self.p = p
            
    def __call__(self, img):
        prob = torch.rand(1).item()
        if prob < self.p[0]:
            return Translation(self.size, self.scale_low_translation, self.scale_high_translation, self.intensity_translation, self.t)(img)
        elif prob < self.p[0] + self.p[1]:
            return Tofro(self.size, self.scale_low_tofro, self.scale_high_tofro, self.intensity_tofro, self.t)(img)
        else:
            return Rotation(self.size, self.scale_low_rotation, self.scale_high_rotation, self.intensity_rotation, self.t)(img)
        
        
class SameHorizontalFlip():
    def __init__(self, p):
        super().__init__()
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            for i in range(len(img)):
                img[i] = F.hflip(img[i])
        return img
    
    
class RandomColorJitter():    
    def __init__(self, p, brightness, contrast, saturation, hue):
        super().__init__()
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        
    def __call__(self, img):
        for i in range(len(img)):
            if torch.rand(1).item() < self.p:
                img[i] = transforms.ColorJitter(
                    brightness=self.brightness, contrast=self.contrast, saturation=self.saturation, hue=self.hue)(img[i])                       
        return img
    
    
class RandomGrayscale():    
    def __init__(self, p):
        super().__init__()
        self.p = p
        
    def __call__(self, img):
        for i in range(len(img)):
            if torch.rand(1).item() < self.p:
                img[i] = F.rgb_to_grayscale(img[i], num_output_channels=3)
        return img
    
    
class RandomSolarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        for i in range(len(img)):
            if torch.rand(1).item() < self.p:
                img[i] = ImageOps.solarize(img[i])
        return img
    
    
class RandomGaussianBlur():
    def __init__(self, p, sigma=[0.1, 2.0]):
        self.p = p
        self.sigma = sigma

    def __call__(self, img):
        for i in range(len(img)):
            if torch.rand(1).item() < self.p:
                sigma = torch.empty(1).uniform_(self.sigma[0], self.sigma[1]).item()
                img[i] = img[i].filter(ImageFilter.GaussianBlur(radius=sigma))
        return img
    
    
class ListtoTensor(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return [transforms.ToTensor()(i) for i in img]
    
    
class ListNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        return [transforms.Normalize(mean=self.mean, std=self.std)(i) for i in img]