from __future__ import print_function, division
import os
import numpy as np
import joblib
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import random
import scipy.ndimage
import math
import SimpleITK as sitk
import pandas as pd
from scipy import ndimage
import sys

slicesize=20
imagesize=[512,512]
cropimagesize=[64,64]
exception=False
MIN_BOUND=-220
MAX_BOUND=280
numclass=2


PatientinfoFile=r'C:\Users\label.xlsx'

class Windowlocate():
    def __call__(self,image,center,width):
        min = (2 * center - width) / 2.0 + 0.5
        max = (2 * center + width) / 2.0 + 0.5
        dFactor = 1024.0 / (max - min)
        image = (image - min)*dFactor
        image[image < min] = min
        image[image > max] = max  # 转换为窗位窗位之后的数据
        return image

class ROINormalization():
    def __call__(self,sample):
        img, msk = sample
        segment=img*msk
        seg_max=segment.max()
        seg_min=segment.min()
        segment=segment/(seg_max-seg_min)
        img=img*(1-msk)/255+segment
        img=np.array(img*255,dtype='uint8') 
        return [img,msk]

class scalerange():
    def __init__(self,rangevalue=[0,1]):
        self.rangevalue=rangevalue
    def __call__(self,sample):
        img, msk = sample
        img=img/255
        return [img,msk]
    
class random_rotate():    
    def __call__(self, sample):
        
        if random.random()<0.5: 
            img,dose=sample          
            rotate_angle = random.randint(-15, 15)
            img = ndimage.rotate(img, rotate_angle, axes=[1,2], reshape=False, mode="nearest", order=1)
            dose= ndimage.rotate(dose, rotate_angle, axes=[1,2], reshape=False, mode="nearest", order=0)
        else:
            img=sample[0]
            dose=sample[1]
            
        return img, dose

class random_filp():
    def __call__(self, sample):

        if random.random()<0.3:
            img,dose=sample
            if random.randint(1) == 1:
                img = np.flip(img, axis=1)
                dose = np.flip(dose, axis=1)
            else:
                img = np.flip(img, axis=0)
                dose = np.flip(dose, axis=0)
        else:

            img=sample[0]
            dose=sample[1]
        return img, dose

class DataAgument():
    def __call__(self, sample):
        img, msk = sample
        msk = msk.transpose((1, 2, 0))

        if False:  # disable left right flip, becuase symmetry object in the image. 
            img = np.flipud(img)
            msk = np.flipud(msk)
        if False:  # disable left right flip, becuase symmetry object in the image. 
            img = np.fliplr(img)
            msk = np.fliplr(msk)
        if random.random() > 0.3:
            angle = random.randint(-20, 20)
            img = scipy.ndimage.rotate(img, angle, reshape=False, order=0, prefilter=False)
            msk = scipy.ndimage.rotate(msk, angle, reshape=False, order=0, prefilter=False)
            # make sure no interpolation as applied in mask
            msk = msk.astype(np.uint8)
            assert np.logical_or(msk==0, msk==1).all()

        msk = msk.transpose((2, 0, 1))
        return [img, msk] 


class ESODataset(Dataset):
    """dataset class for Ultrasound seg task.
    """
    patientinfoFile=PatientinfoFile
    patients_dataframe=pd.DataFrame()
    state_keyname='label'
    save_path='./output3d/'
    def __init__(self, root_dir,savepath,transform=None,image_type='slicer',containtumor=True,istrain=False):
        self.transform = transform
        self.istrain=istrain
        self.exceptlist=[] 
        if savepath==None:
            savepath=''
        pickle_fn = savepath+'.pkl'
        self.root_dir=root_dir
        self.image_type=image_type
        self.containtumor=containtumor
        self.indexloaction=[]

        if os.path.exists(pickle_fn):
            print('load data from {} .....'.format(pickle_fn))
            with open(pickle_fn, 'rb') as f:
                self.img, self.msks,self.patientid,self.patientstate,self.slicercount,self.tumorSlincer= joblib.load(f)
        else:
            self.read_img_mask()  # read all image and mask into memory
            with open(pickle_fn, 'wb') as f:
                if savepath!=None:
                    joblib.dump((self.img, self.msks,self.patientid,self.patientstate,self.slicercount,self.tumorSlincer), f)
                    print('save')
        temp=0
        for x in self.slicercount:
            temp=temp+x
            self.indexloaction.append(temp)

    def resampleVolume(self,outsize,image):
#            """
#        将体数据重采样的指定的spacing大小\n
#        paras：
#        outpacing：指定的spacing，例如[1,1,1]
#        vol：sitk读取的image信息，这里是体数据\n
#        return：重采样后的数据
#        """
        inputspacing = 0
        inputsize = 0
        inputorigin = [0,0,0]
        inputdir = [0,0,0]
        outspacing=[0,0,0]
        #读取文件的size和spacing信息
        
        #读取文件的size和spacing信息
    
        inputsize = image.GetSize()
        inputspacing = image.GetSpacing()
    
        transform = sitk.Transform()
        transform.SetIdentity()
        #计算改变spacing后的size，用物理尺寸/体素的大小
        outspacing[0]=inputsize[0]*inputspacing[0]/outsize[0]
        outspacing[1]=inputsize[1]*inputspacing[1]/outsize[1]
        outspacing[2]=inputsize[2]*inputspacing[2]/outsize[2]
        #设定重采样的一些参数
        resampler = sitk.ResampleImageFilter()
        resampler.SetTransform(transform)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputSpacing(outspacing)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetSize(outsize)
        newvol = resampler.Execute(image)
        return newvol


    def findTumorSlicer(self,msk):
        slicerindex=[]
        size=msk.shape
        for i in range(size[0]):
            if(np.count_nonzero(msk[i,:,:])>1):
                slicerindex.append(i)
        minslice=min(slicerindex)
        maxslice=max(slicerindex)
        return [minslice,maxslice]
    
    def readdataset(self,savepath):
        result=[]
        with open(savepath,'r') as f:
            for line in f.readlines():
                line=line.strip('\n')
                result.append(line)
        return result
    
    def crop_image(self,img,mask):
        size_D=img.shape[0]
        size=math.floor(size_D/2)
        startslice=size-48
        endslice=size+48
        if endslice>size or startslice<0:
            sys.exit

        size_W=img.shape[1]
        size1=math.floor(size_W/2)
        startslice1=size1-48
        endslice1=size1+48
        if endslice1>size_W or startslice1<0:
            sys.exit

        size_H=img.shape[2]
        size2=math.floor(size_H/2)
        startslice2=size2-48
        endslice2=size2+48
        if endslice2>size or startslice2<0:
            sys.exit   

        batchsize=1
        img=img[startslice:endslice,startslice1:endslice1,startslice2:endslice2]
        mask=mask[startslice:endslice,startslice1:endslice1,startslice2:endslice2]

        return img,mask,batchsize
        
    def isinexception(self,path):
        if exception==False:
            return False
        if   self.exceptlist==[] : 
            exception_path=os.path.abspath(os.path.dirname(__file__))+'/exception.txt'
            self.exceptlist=self.readdataset(exception_path)
        return path in self.exceptlist

    def align_coordinate(self,ori_img, target_img, resamplemethod=sitk.sitkNearestNeighbor):
                    
#     "  用itk方法将原始图像resample到与目标图像一致
#    :param ori_img: 原始需要对齐的itk图像
#    :param target_img: 要对齐的目标itk图像
#    :param resamplemethod: itk插值方法: sitk.sitkLinear-线性(CT)  sitk.sitkNearestNeighbor-最近邻(MASK)
#    :return:img_res_itk: 重采样好的itk图像

        target_Size = target_img.GetSize()      # 目标图像大小  [x,y,z]
        target_Spacing = target_img.GetSpacing()   # 目标的体素块尺寸    [x,y,z]
        target_origin = target_img.GetOrigin()      # 目标的起点 [x,y,z]
        target_direction = target_img.GetDirection()  # 目标的方向 [冠,矢,横]=[z,y,x]
        
        # itk的方法进行resample
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ori_img)  # 需要重新采样的目标图像
        # 设置目标图像的信息
        resampler.SetSize(target_Size)		# 目标图像大小
        resampler.SetOutputOrigin(target_origin)
        resampler.SetOutputDirection(target_direction)
        resampler.SetOutputSpacing(target_Spacing)
        # 根据需要重采样图像的情况设置不同的dype
        if resamplemethod == sitk.sitkNearestNeighbor:
            resampler.SetOutputPixelType(sitk.sitkUInt16)   # 近邻插值用于mask的，保存uint16
        else:
            resampler.SetOutputPixelType(sitk.sitkFloat32)  # 线性插值用于PET/CT/MRI之类的，保存float32
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))    
        resampler.SetInterpolator(resamplemethod)
        itk_img_resampled = resampler.Execute(ori_img)  # 得到重新采样后的图像
        return itk_img_resampled
    
    def normalize_HU(self,image):
        image[image > MAX_BOUND] = MAX_BOUND
        image[image < MIN_BOUND] = MIN_BOUND
        # image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        image=(image-image.mean())/image.std()
        return image    
    
    def normalize(self,itk_image):#np.array版本
        min=itk_image.min()
        max=itk_image.max()
        value_range=max-min
        image_array=(itk_image-min)*1.0/value_range
        return image_array  
        
    def read_img_mask(self):
        # get all img and mask filenames
        img_fns = []
        msk_fns = []
        patient_dirs=self.readdataset(self.root_dir)
        for _dir in patient_dirs:
            print('reading patient file names:{}\n'.format(_dir))
            if self.isinexception(_dir)  :
                continue
            tempdir=os.path.basename(_dir) 

            root=r"H:\data"
            img_dir=os.path.join(root,tempdir,'dose.nii.gz')
            msk_dir=os.path.join(root,tempdir,'eso.nii.gz')
          
            assert os.path.isfile(msk_dir), msk_dir
            assert os.path.isfile(img_dir), img_dir    
          
            msk_fns.append(msk_dir)
            img_fns.append(img_dir)

        # read img and mask into memory
        self.imgs = []
        self.msks = []
        self.patientid=[]
        self.slicercount=[]
        self.tumorSlincer=[]
        self.patientstate=[]

        for img_fn, msk_fn in zip(img_fns, msk_fns):
            print('reading img file:{}'.format(img_fn))
            img= sitk.ReadImage(img_fn)
            msk= sitk.ReadImage(msk_fn)

            tempid=os.path.basename(os.path.dirname(img_fn))
            msk=sitk.GetArrayFromImage(msk)
            img=sitk.GetArrayFromImage(img)

            img,msk,batchsize=self.crop_image(img,msk)

            self.imgs.append(img)
            self.msks.append(msk)
            self.patientid.append(tempid)
            self.slicercount.append(batchsize)
            self.patientstate.append(self.__getpatientstate(tempid))

        return 0
   
    def swiftendslice(self,endslice,size):
        halfswiftrange=5
        y=list(range(-halfswiftrange,halfswiftrange))
        random.shuffle(y)
        for i in y:
            temp=endslice+i
            if temp<=size and (temp-slicesize)>=0:
                break
        return temp
    
    def __len__(self):
        if self.image_type=='slicer' and self.containtumor==True:
            length=0
            for i in self.tumorSlincer:
                length=length+len(i)
        elif self.image_type=='slicer' and self.containtumor==False:
            length=sum(self.slicercount)            
        else:            
            length=self.indexloaction[-1]
        return length
    
    def swiftimage(self,img,msk):#####crop 一层图像imagesize到cropimagesize,返回图像，以及截取坐标
        z,x,y=np.nonzero(msk)
        if len(x)==0:
            startH=random.randint(140,260)
            startW=random.randint(140,260)
        else:
            xmin=x.min()
            xmax=x.max()        
            if (xmax-cropimagesize[0])<0:
                temp=0
            else:
                temp=xmax-cropimagesize[0]
            if (imagesize[0]-cropimagesize[0])>xmin:

                startH=random.randint(min(xmin,temp),max(xmin,temp))
            else:
                startH=random.randint(temp,(imagesize[0]-cropimagesize[0]))
            ymin=y.min()
            ymax=y.max()
            if (ymax-cropimagesize[1])<0:
                temp=0
            else:
                temp=ymax-cropimagesize[1]
            if (imagesize[1]-cropimagesize[1])>ymin:
                startW=random.randint(min(temp,ymin),max(temp,ymin))
            else:
                startW=random.randint(temp,(imagesize[1]-cropimagesize[1]))
        

        img=img[:,startH:(startH+cropimagesize[0]),startW:(startW+cropimagesize[1])]
        msk=msk[:,startH:(startH+cropimagesize[0]),startW:(startW+cropimagesize[1])]
        return img,msk,np.array([startH,startW])
    
    def cropimageXY(self,img,msk):
        z,x,y=np.nonzero(msk)
        if len(x)==0:
            startH=random.randint(120,260)
            startW=random.randint(170,330)
        else:
            xmin=x.min()
            xmax=x.max()        
            startH=int((xmax+xmin)/2-cropimagesize[0]/2)
            ymin=y.min()
            ymax=y.max()
            startW=int((ymax+ymin)/2-cropimagesize[1]/2)
        
        img=img[:,startH:(startH+cropimagesize[0]),startW:(startW+cropimagesize[1])]
        msk=msk[:,startH:(startH+cropimagesize[0]),startW:(startW+cropimagesize[1])]
        return img,msk,np.array([startH,startW])

    def __getpatientstate(self,patientid):
        if self.patients_dataframe.empty:
            self.patients_dataframe=pd.DataFrame(pd.read_excel(self.patientinfoFile))
            self.patients_dataframe['patientname']=self.patients_dataframe['patientname'].apply(str) 
        state=self.patients_dataframe[self.patients_dataframe.patientname==patientid][self.state_keyname]
        if len(state.values)==0 or math.isnan(state.values[0]):
            state=None
            print("state wrong!!!!")
        else:
            state=state.values[0]
        return state
    
    def __getitem__(self, idx):
       
        if(self.istrain==True):
            indexlocation=next(x for x in self.indexloaction if x >idx)
            imgidx=self.indexloaction.index(indexlocation)
            img=self.imgs[imgidx]
            msk=self.msks[imgidx]
            patientid=self.patientid[imgidx]
            patientstate=self.patientstate[imgidx]
            
            if self.transform:
                compose, msk = self.transform([compose, msk])
            compose,msk,location= self.cropimageXY(compose, msk)
        else:
            indexlocation=next(x for x in self.indexloaction if x >idx)
            imgidx=self.indexloaction.index(indexlocation)            
            img=self.imgs[imgidx]           
            msk=self.msks[imgidx]
            patientid=self.patientid[imgidx]
            patientstate=self.patientstate[imgidx]
            
        img = torch.from_numpy(img.astype(float))
        img=img.float().unsqueeze(0)
        
        msk=np.uint8(msk>0)
        msk=np.expand_dims(msk, axis=0)

        return [img,msk,patientid,patientstate]


def backcrop(img,msk,location):
    [startH,startW]=location
    size=img.shape
    tempimg=np.zeros([size[0],imagesize[0],imagesize[1]])
    tempmsk=np.zeros([size[0],imagesize[0],imagesize[1]])
    tempimg[:,startH:(startH+cropimagesize[0]),startW:(startW+cropimagesize[1])]=img
    tempmsk[:,startH:(startH+cropimagesize[0]),startW:(startW+cropimagesize[1])]=msk
    return tempimg,tempmsk

def get_dataloaders(batch_size,containtumor=False):
    trans = transforms.Compose([])
    trans1= transforms.Compose([ROINormalization(),scalerange()])
    trans2 = transforms.Compose([random_filp(),random_rotate()])



    root_dir=os.path.abspath('.')+'/train.txt'
    savepath=os.path.abspath('.')+r'/output3d/train'
    imgtype='random'
    containtumor=False
    
    trainsets = ESODataset(root_dir,savepath,trans,imgtype,containtumor,istrain=True)
    
    root_dir=os.path.abspath('.')+'/val.txt'
    savepath=os.path.abspath('.')+r'/output3d/val'
    valsets = ESODataset(root_dir,savepath,trans,imgtype,containtumor,istrain=False)

    imgtype='all'
    root_dir=os.path.abspath('.')+'/test.txt'
    savepath=os.path.abspath('.')+r'/output3d/test'

    testsets = ESODataset(root_dir,savepath,trans,imgtype,containtumor,istrain=False)    
   
    dataloaders = {
        'train': DataLoader(trainsets, batch_size=batch_size, shuffle=True, num_workers=0),
        'val':  DataLoader(valsets, batch_size=batch_size, shuffle=False, num_workers=0),
        'test': DataLoader(testsets, batch_size=batch_size, shuffle=False, num_workers=0)
    }

    dataset_sizes = {
        'train': len(trainsets),
        'val':  len(valsets),
        'test': len(testsets)
    }
    
    return dataloaders, dataset_sizes


if __name__ == "__main__":
    dataloaders, size = get_dataloaders(4)  
    print(os.path.abspath(os.path.dirname(__file__)+'/dataloader/'))

