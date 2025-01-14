import torch.utils.data as data
import cv2
import torch
import numpy as np
import math
from .draw_gaussian import draw_umich_gaussian, gaussian_radius
from .transforms import random_flip, load_affine_matrix, random_crop_info, ex_box_jaccard, random_grid_erase
from . import data_augment

class BaseDataset(data.Dataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, down_ratio=None):
        super(BaseDataset, self).__init__()
        self.data_dir = data_dir
        self.phase = phase
        self.input_h = input_h
        self.input_w = input_w
        self.down_ratio = down_ratio
        self.img_ids = None
        self.num_classes = None
        self.max_objs = 500
        self.image_distort =  data_augment.PhotometricDistort()

    def load_img_ids(self):
        """
        Definition: generate self.img_ids
        Usage: index the image properties (e.g. image name) for training, testing and evaluation
        Format: self.img_ids = [list]
        Return: self.img_ids
        """
        return None

    def load_image(self, index):
        """
        Definition: read images online
        Input: index, the index of the image in self.img_ids
        Return: image with H x W x 3 format
        """
        return None

    def load_annoFolder(self, img_id):
        """
        Return: the path of annotation
        Note: You may not need this function
        """
        return None

    def load_annotation(self, index):
        """
        Return: dictionary of {'pts': float np array of [bl, tl, tr, br], 
                                'cat': int np array of class_index}
        Explaination:
                bl: bottom left point of the bounding box, format [x, y]
                tl: top left point of the bounding box, format [x, y]
                tr: top right point of the bounding box, format [x, y]
                br: bottom right point of the bounding box, format [x, y]
                class_index: the category index in self.category
                    example: self.category = ['ship]
                             class_index of ship = 0
        """
        return None

    def dec_evaluation(self, result_path):
        return None

    def data_transform(self, image, annotation):
        # only do random_flip augmentation to original images
        crop_size = None
        crop_center = None
        crop_size, crop_center = random_crop_info(h=image.shape[0], w=image.shape[1])
        #image, annotation['pts'], crop_center = random_flip(image, annotation['pts'], crop_center)  # 对于标志标线，不应该有flip类型的对称翻转
        #image, annotation['pts'] = random_grid_erase(image, annotation['pts'])    # grid erase, add by mixiaoxin

        if crop_center is None:
            crop_center = np.asarray([float(image.shape[1])/2, float(image.shape[0])/2], dtype=np.float32)
        if crop_size is None:
            crop_size = [max(image.shape[1], image.shape[0]), max(image.shape[1], image.shape[0])]  # init
        M = load_affine_matrix(crop_center=crop_center,
                               crop_size=crop_size,
                               dst_size=(self.input_w, self.input_h),
                               inverse=False,
                               rotation=True)
        image = cv2.warpAffine(src=image, M=M, dsize=(self.input_w, self.input_h), flags=cv2.INTER_LINEAR) # 实现图像的平移、旋转
        if annotation['pts'].shape[0]:  # 4 个 线程，一次load4张image，并对图像裁剪等放射变换
            #print('annotation pts size 1: ', annotation['pts'].shape)
            annotation['pts'] = np.concatenate([annotation['pts'], np.ones((annotation['pts'].shape[0], annotation['pts'].shape[1], 1))], axis=2)  # 扩充一位，以便下方的矩阵运算
            #print('annotation pts size 2: ', annotation['pts'].shape)
            annotation['pts'] = np.matmul(annotation['pts'], np.transpose(M))   # 平移、旋转变换
            #print('annotation pts size 3: ', annotation['pts'].shape)
            annotation['pts'] = np.asarray(annotation['pts'], np.float32)

        out_annotations = {}
        size_thresh = 0.5
        out_rects = []
        out_cat = []
        for pt_old, cat in zip(annotation['pts'] , annotation['cat']):
            if (pt_old<0).any() or (pt_old[:,0]>self.input_w-1).any() or (pt_old[:,1]>self.input_h-1).any(): # 若变换后的图像超出预定义尺寸
                pt_new = pt_old.copy()
                pt_new[:,0] = np.minimum(np.maximum(pt_new[:,0], 0.), self.input_w - 1)
                pt_new[:,1] = np.minimum(np.maximum(pt_new[:,1], 0.), self.input_h - 1)
                iou = ex_box_jaccard(pt_old.copy(), pt_new.copy())
                if iou>0.6:
                    ############### origin ##############################################################
                    #rect = cv2.minAreaRect(pt_new/self.down_ratio)  # 这个函数计算出来后第一个点的标注信息就被抹去了
                    # if rect[1][0]>size_thresh and rect[1][1]>size_thresh:
                    #     out_rects.append([rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]])
                    #     out_cat.append(cat)
                    #######################################################################################
                    #### change  by mixiaoxin #############################################################
                    # (Done 2021-12-02)TODO：： change the rect annotation as 4 vertexes but width 、 height 、angle
                    rect = pt_new / self.down_ratio    # change by mixiaoxin
                    rect_edge1 = np.sqrt(np.sum(np.square(rect[0, :] - rect[1, :])))
                    rect_edge2 = np.sqrt(np.sum(np.square(rect[0, :] - rect[3, :])))
                    if rect_edge1>size_thresh and rect_edge2>size_thresh:
                        out_rects.append(rect)
                        out_cat.append(cat)
                    #######################################################################################
            else:
                ############### origin ##############################################################
                #rect = cv2.minAreaRect(pt_old/self.down_ratio)
                # if rect[1][0]<size_thresh and rect[1][1]<size_thresh:
                #     continue
                # out_rects.append([rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]])
                # out_cat.append(cat)
                #######################################################################################
                #### change  by mixiaoxin #############################################################
                rect = pt_old / self.down_ratio  # change by mixiaoxin
                rect_edge1 = np.sqrt(np.sum(np.square(rect[0, :] - rect[1, :])))
                rect_edge2 = np.sqrt(np.sum(np.square(rect[0, :] - rect[3, :])))
                if rect_edge1<size_thresh or rect_edge2<size_thresh:
                    continue
                out_rects.append(rect)
                out_cat.append(cat)
                #######################################################################################
        out_annotations['rect'] = np.asarray(out_rects, np.float32)
        out_annotations['cat'] = np.asarray(out_cat, np.uint8)
        return image, out_annotations

    def __len__(self):
        return len(self.img_ids)

    def processing_test(self, image, input_h, input_w):
        image = cv2.resize(image, (input_w, input_h))
        out_image = image.astype(np.float32) / 255.
        out_image = out_image - 0.5
        out_image = out_image.transpose(2, 0, 1).reshape(1, 3, input_h, input_w)
        out_image = torch.from_numpy(out_image)
        return out_image

    def cal_bbox_wh(self, pts_4):
        x1 = np.min(pts_4[:,0])
        x2 = np.max(pts_4[:,0])
        y1 = np.min(pts_4[:,1])
        y2 = np.max(pts_4[:,1])
        return x2-x1, y2-y1


    def cal_bbox_pts(self, pts_4):
        x1 = np.min(pts_4[:,0])
        x2 = np.max(pts_4[:,0])
        y1 = np.min(pts_4[:,1])
        y2 = np.max(pts_4[:,1])
        bl = [x1, y2]
        tl = [x1, y1]
        tr = [x2, y1]
        br = [x2, y2]
        return np.asarray([bl, tl, tr, br], np.float32)

    def reorder_pts(self, tt, rr, bb, ll):
        pts = np.asarray([tt,rr,bb,ll],np.float32)
        l_ind = np.argmin(pts[:,0])
        r_ind = np.argmax(pts[:,0])
        t_ind = np.argmin(pts[:,1])
        b_ind = np.argmax(pts[:,1])
        tt_new = pts[t_ind,:]
        rr_new = pts[r_ind,:]
        bb_new = pts[b_ind,:]
        ll_new = pts[l_ind,:]
        return tt_new,rr_new,bb_new,ll_new


    def generate_ground_truth(self, image, annotation):
        image = np.asarray(np.clip(image, a_min=0., a_max=255.), np.float32)  # 强度限制至0~255
        image = self.image_distort(np.asarray(image, np.float32))             # 强度纠正，包括：增强对比度、增强亮度、去躁
        image = np.asarray(np.clip(image, a_min=0., a_max=255.), np.float32)
        # ###################################### view Images #######################################
        # image_source = image.copy()
        # ##########################################################################################
        image = np.transpose(image / 255. - 0.5, (2, 0, 1))                   # 将读入的BGR通道转为RGB通道

        image_h = self.input_h // self.down_ratio                             # 图像降采样，向下取整
        image_w = self.input_w // self.down_ratio

        hm = np.zeros((self.num_classes, image_h, image_w), dtype=np.float32) # heat map的尺寸：
        wh = np.zeros((self.max_objs, 10), dtype=np.float32)                  # oriented bounding box的回归参数矩阵： maxobjects × 10 （每个bounding box对应10个参数）
        ## add
        cls_theta = np.zeros((self.max_objs, 1), dtype=np.float32)            # rotated box 与 horizontal box的分类分支，尺寸为max objects × 1
        ## add end
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)                  # center point浮点数回归参数： max-objs × 2 （2表示x 或者 y）
        ind = np.zeros((self.max_objs), dtype=np.int64)                       # ind为box的索引id，数量为500个， TODO： 后期这个参数是不是可以改小一点呢？会对整体效果有什么影响呢？
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)                  # reg_mask 是什么呢？ 尺寸为500的无符号整数 ？？？
        num_objs = min(annotation['rect'].shape[0], self.max_objs)
        # ###################################### view Images #######################################
        # print('image: ', image.shape)
        # copy_image1 = cv2.resize(image_source, (image_w, image_h))
        # copy_image2 = copy_image1.copy()
        # ##########################################################################################
        for k in range(num_objs):
            rect = annotation['rect'][k, :]
            cen_x, cen_y, bbox_w, bbox_h, theta = rect # 此时的theta为计算的最小包围矩形的第一个顶点，沿着u轴逆时针旋转遇到第一条边的角度，取值为【0， -90°】
            # print(theta)
            radius = gaussian_radius((math.ceil(bbox_h), math.ceil(bbox_w)))
            radius = max(0, int(radius))
            ct = np.asarray([cen_x, cen_y], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(hm[annotation['cat'][k]], ct_int, radius)
            ind[k] = ct_int[1] * image_w + ct_int[0]
            reg[k] = ct - ct_int  # offset的浮点
            reg_mask[k] = 1
            # generate wh ground_truth
            pts_4 = cv2.boxPoints(((cen_x, cen_y), (bbox_w, bbox_h), theta))  # 4 x 2

            bl = pts_4[0,:]
            tl = pts_4[1,:]
            tr = pts_4[2,:]
            br = pts_4[3,:]

            tt = (np.asarray(tl,np.float32)+np.asarray(tr,np.float32))/2
            rr = (np.asarray(tr,np.float32)+np.asarray(br,np.float32))/2
            bb = (np.asarray(bl,np.float32)+np.asarray(br,np.float32))/2
            ll = (np.asarray(tl,np.float32)+np.asarray(bl,np.float32))/2

            if theta in [-90.0, -0.0, 0.0]:  # (-90, 0]
                tt,rr,bb,ll = self.reorder_pts(tt,rr,bb,ll)
            # rotational channel
            wh[k, 0:2] = tt - ct
            wh[k, 2:4] = rr - ct
            wh[k, 4:6] = bb - ct
            wh[k, 6:8] = ll - ct
            #####################################################################################
            # # draw obb
            # cv2.line(copy_image1, (int(cen_x), int(cen_y)), (int(tt[0]), int(tt[1])), (0, 0, 255), 1, 1)
            # cv2.line(copy_image1, (int(cen_x), int(cen_y)), (int(rr[0]), int(rr[1])), (255, 0, 255), 1, 1)
            # cv2.line(copy_image1, (int(cen_x), int(cen_y)), (int(bb[0]), int(bb[1])), (0, 255, 255), 1, 1)
            # cv2.line(copy_image1, (int(cen_x), int(cen_y)), (int(ll[0]), int(ll[1])), (255, 0, 0), 1, 1)
            ####################################################################################
            # horizontal channel
            w_hbbox, h_hbbox = self.cal_bbox_wh(pts_4)
            wh[k, 8:10] = 1. * w_hbbox, 1. * h_hbbox
            #####################################################################################
            # # draw hbb
            # cv2.line(copy_image2, (int(cen_x), int(cen_y)), (int(cen_x), int(cen_y-wh[k, 9]/2)), (0, 0, 255), 1, 1)
            # cv2.line(copy_image2, (int(cen_x), int(cen_y)), (int(cen_x+wh[k, 8]/2), int(cen_y)), (255, 0, 255), 1, 1)
            # cv2.line(copy_image2, (int(cen_x), int(cen_y)), (int(cen_x), int(cen_y+wh[k, 9]/2)), (0, 255, 255), 1, 1)
            # cv2.line(copy_image2, (int(cen_x), int(cen_y)), (int(cen_x-wh[k, 8]/2), int(cen_y)), (255, 0, 0), 1, 1)
            #####################################################################################
            # v0
            # if abs(theta)>3 and abs(theta)<90-3:
            #     cls_theta[k, 0] = 1
            # v1
            jaccard_score = ex_box_jaccard(pts_4.copy(), self.cal_bbox_pts(pts_4).copy())
            if jaccard_score<0.95:
                cls_theta[k, 0] = 1


        # ###################################### view Images #####################################
        # hm_show = np.uint8(cv2.applyColorMap(np.uint8(hm[0, :, :] * 255), cv2.COLORMAP_JET))
        # copy_image = cv2.addWeighted(np.uint8(copy_image2), 0.4, hm_show, 0.8, 0)
        # cv2.imshow('img-heatmap', cv2.resize(np.uint8(copy_image), (image_w * 4, image_h * 4)))
        # key_heatmap = cv2.waitKey(0) & 0xFF
        #
        # if jaccard_score>0.95:
        #     print(theta, jaccard_score, cls_theta[k, 0])
        #     cv2.imshow('img1', cv2.resize(np.uint8(copy_image1), (image_w*4, image_h*4)))
        #     cv2.imshow('img2', cv2.resize(np.uint8(copy_image2), (image_w*4, image_h*4)))
        #     key = cv2.waitKey(0)&0xFF
        #     if key==ord('q'):
        #         cv2.destroyAllWindows()
        #         exit()
        # #########################################################################################

        ret = {'input': image,
               'hm': hm,
               'reg_mask': reg_mask,
               'ind': ind,
               'wh': wh,
               'reg': reg,
               'cls_theta':cls_theta,
               }
        return ret

    def generate_ground_truth_roadmarking(self, image, annotation):
        image = np.asarray(np.clip(image, a_min=0., a_max=255.), np.float32)  # 强度限制至0~255
        image = self.image_distort(np.asarray(image, np.float32))             # 强度纠正，包括：增强对比度、增强亮度、去躁
        image = np.asarray(np.clip(image, a_min=0., a_max=255.), np.float32)
        # ###################################### view Images #######################################
        # image_source = image.copy()
        # ##########################################################################################
        image = np.transpose(image / 255. - 0.5, (2, 0, 1))                   # 将读入的图像转为channel × height × width

        image_h = self.input_h // self.down_ratio                             # 图像降采样，向下取整
        image_w = self.input_w // self.down_ratio

        hm = np.zeros((self.num_classes, image_h, image_w), dtype=np.float32) # heat map的尺寸：
        wh = np.zeros((self.max_objs, 10), dtype=np.float32)                  # oriented bounding box的回归参数矩阵： maxobjects × 10 （每个bounding box对应10个参数）
        ## add
        cls_theta = np.zeros((self.max_objs, 1), dtype=np.float32)            # rotated box 与 horizontal box的分类分支，尺寸为max objects × 1
        ## add end
        ## add forward direction mixiaoxin
        forward = np.zeros((self.max_objs, 1), dtype=np.float32)              # forward 方向的分类分支，尺寸为max objects × 1，朝向v轴正方向为1，否则为0
        forward_mask = np.zeros((self.max_objs), dtype=np.uint8)              # 每个位置上是否需要需要预测方向
        ## add end
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)                  # center point浮点数回归参数： max-objs × 2 （2表示x 或者 y）
        ind = np.zeros((self.max_objs), dtype=np.int64)                       # ind为box在缩放后的图像上的位置索引id，数量为500个， TODO： 后期这个参数是不是可以改小一点呢？会对整体效果有什么影响呢？
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)                  # reg_mask 是什么呢？ 尺寸为500的无符号整数, 每个位置上是否有ground truth的mask，有则为1，无则为0

        num_objs = min(annotation['rect'].shape[0], self.max_objs)
        # ###################################### view Images #######################################
        # print('image source: ', image_source.shape)
        # print('image: ', image.shape)
        # copy_image1 = cv2.resize(image_source, (image_w, image_h))
        # copy_image2 = copy_image1.copy()
        # ##########################################################################################
        for k in range(num_objs):
            rect = annotation['rect'][k, :]
            # TODO： reorgonize the center and bbox annotation accoring to the new rect style by mixiaoxin
            pts_4 = rect
            #cen_x, cen_y, bbox_w, bbox_h, theta = rect
            # print(theta)
            bbox_w = np.sqrt(np.sum(np.square(np.asarray(pts_4[0, :] - pts_4[1, :], np.float32))))
            bbox_h = np.sqrt(np.sum(np.square(np.asarray(pts_4[0, :] - pts_4[3, :], np.float32))))
            #print('obb width is {}, obb height is {};'.format(bbox_w, bbox_h))
            cen_x, cen_y = pts_4[0, :] * 0.5 + pts_4[2, :] * 0.5
            radius = gaussian_radius((math.ceil(bbox_h), math.ceil(bbox_w)))
            radius = max(0, int(radius))
            ct = np.asarray([cen_x, cen_y], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(hm[annotation['cat'][k]], ct_int, radius)
            ind[k] = ct_int[1] * image_w + ct_int[0]
            reg[k] = ct - ct_int  # offset的浮点
            reg_mask[k] = 1
            forward_mask[k] = 1
            # generate wh ground_truth
            tl = pts_4[0,:]
            tr = pts_4[1,:]
            br = pts_4[2,:]
            bl = pts_4[3,:]

            # bl = pts_4[0, :]
            # tl = pts_4[1, :]
            # tr = pts_4[2, :]
            # br = pts_4[3, :]

            tt = (np.asarray(tl,np.float32)+np.asarray(tr,np.float32))/2
            rr = (np.asarray(tr,np.float32)+np.asarray(br,np.float32))/2
            bb = (np.asarray(bl,np.float32)+np.asarray(br,np.float32))/2
            ll = (np.asarray(tl,np.float32)+np.asarray(bl,np.float32))/2
            tt, rr, bb, ll = self.reorder_pts(tt, rr, bb, ll)

            # rotational channel
            wh[k, 0:2] = tt - ct
            wh[k, 2:4] = rr - ct
            wh[k, 4:6] = bb - ct
            wh[k, 6:8] = ll - ct
            #####################################################################################
            # # draw
            # cv2.line(copy_image1, (int(cen_x), int(cen_y)), (int(tt[0]), int(tt[1])), (0, 0, 255), 1, 1)
            # cv2.line(copy_image1, (int(cen_x), int(cen_y)), (int(rr[0]), int(rr[1])), (255, 0, 255), 1, 1)
            # cv2.line(copy_image1, (int(cen_x), int(cen_y)), (int(bb[0]), int(bb[1])), (0, 255, 255), 1, 1)
            # cv2.line(copy_image1, (int(cen_x), int(cen_y)), (int(ll[0]), int(ll[1])), (255, 0, 0), 1, 1)
            # ####################################################################################
            # horizontal channel
            w_hbbox, h_hbbox = self.cal_bbox_wh(pts_4)
            wh[k, 8:10] = 1. * w_hbbox, 1. * h_hbbox
            #####################################################################################
            # # draw
            # cv2.line(copy_image2, (int(cen_x), int(cen_y)), (int(cen_x), int(cen_y-wh[k, 9]/2)), (0, 0, 255), 1, 1)
            # cv2.line(copy_image2, (int(cen_x), int(cen_y)), (int(cen_x+wh[k, 8]/2), int(cen_y)), (255, 0, 255), 1, 1)
            # cv2.line(copy_image2, (int(cen_x), int(cen_y)), (int(cen_x), int(cen_y+wh[k, 9]/2)), (0, 255, 255), 1, 1)
            # cv2.line(copy_image2, (int(cen_x), int(cen_y)), (int(cen_x-wh[k, 8]/2), int(cen_y)), (255, 0, 0), 1, 1)
            #####################################################################################
            # v0
            # if abs(theta)>3 and abs(theta)<90-3:
            #     cls_theta[k, 0] = 1
            # v1
            jaccard_score = ex_box_jaccard(pts_4.copy(), self.cal_bbox_pts(pts_4).copy())
            if jaccard_score<0.95:
                cls_theta[k, 0] = 1
            forward_symbol = pts_4[0,1] - pts_4[2,1]   # v 轴，height方向
            if forward_symbol > 0:
                forward[k, 0] = 1
            ## TODO: 是否要在这里增加线状地物的方向监督为2呢？这些类别不具有正方向的特性，还是在计算预测结果的时候计算呢？
            #print("this category is: ", annotation['cat'][k])
            # if annotation['cat'][k] == 'Forbidden' or annotation['cat'][k] == 'Diamond' or annotation['cat'][k] == 'Dashed_lane' \
            #     or annotation['cat'][k] == 'Zebra_crossing' or annotation['cat'][k] == 'Stop_lane':
            if annotation['cat'][k] == 6 or annotation['cat'][k] == 7 or annotation['cat'][k] == 10 \
                        or annotation['cat'][k] == 11 or annotation['cat'][k] == 12:
                forward[k, 0] = 2
                forward_mask[k] = 0 # 这几个类别不预测正方向

        # ###################################### view Images #####################################
            # hm_show = np.uint8(cv2.applyColorMap(np.uint8(hm[0, :, :] * 255), cv2.COLORMAP_JET))
            # copy_image = cv2.addWeighted(np.uint8(copy_image2), 0.4, hm_show, 0.8, 0)
            # cv2.imshow('img-heatmap', cv2.resize(np.uint8(copy_image), (image_w * 4, image_h * 4)))
            # key_heatmap = cv2.waitKey(0) & 0xFF
            # if forward[k, 0] > 0.5:
            #     cv2.circle(copy_image1, (int(bb[0]), int(bb[1])), 2, (255, 255, 0), 4)
            # else:
            #     cv2.circle(copy_image1, (int(tt[0]), int(tt[1])), 2, (255, 255, 0), 4)
            # if jaccard_score>0.95:
            #     #print(theta, jaccard_score, cls_theta[k, 0])
            #     cv2.imshow('img1', cv2.resize(np.uint8(copy_image1), (image_w*4, image_h*4)))
            #     cv2.imshow('img2', cv2.resize(np.uint8(copy_image2), (image_w*4, image_h*4)))
            #     key = cv2.waitKey(0)&0xFF
            #     if key==ord('q'):
            #         cv2.destroyAllWindows()
            #         exit()
        # #########################################################################################
        # print('number of objects: ', num_objs)
        # print('forward: ', forward)
        # print('forward mask: ', forward_mask)

        ret = {'input': image,
               'hm': hm,
               'reg_mask': reg_mask,
               'ind': ind,
               'wh': wh,
               'reg': reg,
               'cls_theta':cls_theta,
               'forward':forward,
               'forward_mask': forward_mask,
               }
        return ret

    def generate_ground_truth_roadmarking(self, image, annotation):
        image = np.asarray(np.clip(image, a_min=0., a_max=255.), np.float32)  # 强度限制至0~255
        image = self.image_distort(np.asarray(image, np.float32))             # 强度纠正，包括：增强对比度、增强亮度、去躁
        image = np.asarray(np.clip(image, a_min=0., a_max=255.), np.float32)
        # ###################################### view Images #######################################
        #image_source = image.copy()
        # ##########################################################################################
        image = np.transpose(image / 255. - 0.5, (2, 0, 1))                   # 将读入的BGR通道转为RGB通道

        image_h = self.input_h // self.down_ratio                             # 图像降采样，向下取整
        image_w = self.input_w // self.down_ratio

        hm = np.zeros((self.num_classes, image_h, image_w), dtype=np.float32) # heat map的尺寸：
        wh = np.zeros((self.max_objs, 10), dtype=np.float32)                  # oriented bounding box的回归参数矩阵： maxobjects × 10 （每个bounding box对应10个参数）
        ## add
        cls_theta = np.zeros((self.max_objs, 1), dtype=np.float32)            # rotated box 与 horizontal box的分类分支，尺寸为max objects × 1
        ## add end
        ## add forward direction mixiaoxin
        forward = np.zeros((self.max_objs, 1), dtype=np.float32)              # forward 方向的分类分支，尺寸为max objects × 1，朝向v轴正方向为1，否则为0
        ## add end
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)                  # center point浮点数回归参数： max-objs × 2 （2表示x 或者 y）
        ind = np.zeros((self.max_objs), dtype=np.int64)                       # ind为box的索引id，数量为500个， TODO： 后期这个参数是不是可以改小一点呢？会对整体效果有什么影响呢？
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)                  # reg_mask 是什么呢？ 尺寸为500的无符号整数 ？？？
        num_objs = min(annotation['rect'].shape[0], self.max_objs)
        # ###################################### view Images #######################################
        # print('image: ', image.shape)
        # copy_image1 = cv2.resize(image_source, (image_w, image_h))
        # copy_image2 = copy_image1.copy()
        # ##########################################################################################
        for k in range(num_objs):
            rect = annotation['rect'][k, :]
            # TODO： reorgonize the center and bbox annotation accoring to the new rect style by mixiaoxin
            pts_4 = rect
            #cen_x, cen_y, bbox_w, bbox_h, theta = rect
            # print(theta)
            bbox_w = np.sqrt(np.sum(np.square(np.asarray(pts_4[0, :] - pts_4[1, :], np.float32))))
            bbox_h = np.sqrt(np.sum(np.square(np.asarray(pts_4[0, :] - pts_4[3, :], np.float32))))
            cen_x, cen_y = pts_4[0, :] * 0.5 + pts_4[2, :] * 0.5
            radius = gaussian_radius((math.ceil(bbox_h), math.ceil(bbox_w)))
            radius = max(0, int(radius))
            ct = np.asarray([cen_x, cen_y], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(hm[annotation['cat'][k]], ct_int, radius)
            ind[k] = ct_int[1] * image_w + ct_int[0]
            reg[k] = ct - ct_int  # offset的浮点
            reg_mask[k] = 1
            # generate wh ground_truth
            bl = pts_4[0,:]
            tl = pts_4[1,:]
            tr = pts_4[2,:]
            br = pts_4[3,:]

            tt = (np.asarray(tl,np.float32)+np.asarray(tr,np.float32))/2
            rr = (np.asarray(tr,np.float32)+np.asarray(br,np.float32))/2
            bb = (np.asarray(bl,np.float32)+np.asarray(br,np.float32))/2
            ll = (np.asarray(tl,np.float32)+np.asarray(bl,np.float32))/2
            tt, rr, bb, ll = self.reorder_pts(tt, rr, bb, ll)

            # rotational channel
            wh[k, 0:2] = tt - ct
            wh[k, 2:4] = rr - ct
            wh[k, 4:6] = bb - ct
            wh[k, 6:8] = ll - ct
            #####################################################################################
            # # draw
            # cv2.line(copy_image1, (int(cen_x), int(cen_y)), (int(tt[0]), int(tt[1])), (0, 0, 255), 1, 1)
            # cv2.line(copy_image1, (int(cen_x), int(cen_y)), (int(rr[0]), int(rr[1])), (255, 0, 255), 1, 1)
            # cv2.line(copy_image1, (int(cen_x), int(cen_y)), (int(bb[0]), int(bb[1])), (0, 255, 255), 1, 1)
            # cv2.line(copy_image1, (int(cen_x), int(cen_y)), (int(ll[0]), int(ll[1])), (255, 0, 0), 1, 1)
            ####################################################################################
            # horizontal channel
            w_hbbox, h_hbbox = self.cal_bbox_wh(pts_4)
            wh[k, 8:10] = 1. * w_hbbox, 1. * h_hbbox
            #####################################################################################
            # # draw
            # cv2.line(copy_image2, (int(cen_x), int(cen_y)), (int(cen_x), int(cen_y-wh[k, 9]/2)), (0, 0, 255), 1, 1)
            # cv2.line(copy_image2, (int(cen_x), int(cen_y)), (int(cen_x+wh[k, 8]/2), int(cen_y)), (255, 0, 255), 1, 1)
            # cv2.line(copy_image2, (int(cen_x), int(cen_y)), (int(cen_x), int(cen_y+wh[k, 9]/2)), (0, 255, 255), 1, 1)
            # cv2.line(copy_image2, (int(cen_x), int(cen_y)), (int(cen_x-wh[k, 8]/2), int(cen_y)), (255, 0, 0), 1, 1)
            #####################################################################################
            # v0
            # if abs(theta)>3 and abs(theta)<90-3:
            #     cls_theta[k, 0] = 1
            # v1
            jaccard_score = ex_box_jaccard(pts_4.copy(), self.cal_bbox_pts(pts_4).copy())
            if jaccard_score<0.95:
                cls_theta[k, 0] = 1

            forward_symbol = pts_4[0,1] - pts_4[2,1]   # v 轴，height方向
            if forward_symbol > 0:
                forward[k, 0] = 1
            ## TODO: 是否要在这里增加线状地物的方向监督为2呢？这些类别不具有正方向的特性，还是在计算预测结果的时候计算呢？
            if annotation['cat'][k] == 'Forbidden' or annotation['cat'][k] == 'Diamond' or annotation['cat'][k] == 'Dashed_lane' \
                or annotation['cat'][k] == 'Zebra_crossing' or annotation['cat'][k] == 'Stop_lane':
                forward[k, 0] = 2

        # ###################################### view Images #####################################
        # hm_show = np.uint8(cv2.applyColorMap(np.uint8(hm[0, :, :] * 255), cv2.COLORMAP_JET))
        # copy_image = cv2.addWeighted(np.uint8(copy_image2), 0.4, hm_show, 0.8, 0)
        #     if jaccard_score>0.95:
        #         print(theta, jaccard_score, cls_theta[k, 0])
        #         cv2.imshow('img1', cv2.resize(np.uint8(copy_image1), (image_w*4, image_h*4)))
        #         cv2.imshow('img2', cv2.resize(np.uint8(copy_image2), (image_w*4, image_h*4)))
        #         key = cv2.waitKey(0)&0xFF
        #         if key==ord('q'):
        #             cv2.destroyAllWindows()
        #             exit()
        # #########################################################################################

        ret = {'input': image,
               'hm': hm,
               'reg_mask': reg_mask,
               'ind': ind,
               'wh': wh,
               'reg': reg,
               'cls_theta':cls_theta,
               'forward':forward,
               }
        return ret

    def __getitem__(self, index):
        image = self.load_image(index)
        image_h, image_w, c = image.shape
        if self.phase == 'test':
            img_id = self.img_ids[index]
            image = self.processing_test(image, self.input_h, self.input_w)
            return {'image': image,
                    'img_id': img_id,
                    'image_w': image_w,
                    'image_h': image_h}

        elif self.phase == 'train':
            annotation = self.load_annotation(index)
            image, annotation = self.data_transform(image, annotation)
            #data_dict = self.generate_ground_truth(image, annotation)
            data_dict = self.generate_ground_truth_roadmarking(image, annotation) # modify by mixiaoxin to fit the application
            return data_dict


