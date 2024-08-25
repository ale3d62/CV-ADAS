
import torch
from utils import *


def detect(img, model, device, imgsz=640):
    #inf_time = AverageMeter()
    #waste_time = AverageMeter()
    #nms_time = AverageMeter()

    # Load model
    stride =32

    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = model.to(device)

    if half:
        model.half()  # to FP16  
    model.eval()

    # Set Dataloader
    #dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    #t0 = time.time()
    #for path, img, im0s, vid_cap in dataset:
    


    #img = cv2.resize(img, (1280,720), interpolation=cv2.INTER_LINEAR)
    img = letterbox(img)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)



    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    #img = img.permute(0, 3, 1, 2)
    # Inference
    #t1 = time_synchronized()

    [pred,anchor_grid],seg,ll= model(img)
    #t2 = time_synchronized()

    # waste time: the incompatibility of  torch.jit.trace causes extra time consumption in demo version 
    # but this problem will not appear in offical version 
    #tw1 = time_synchronized()
    pred = split_for_trace_model(pred,anchor_grid)
    #tw2 = time_synchronized()

    # Apply NMS
    #t3 = time_synchronized()
    pred = non_max_suppression(pred)
    #t4 = time_synchronized()

    #da_seg_mask = driving_area_mask(seg)
    ll_seg_mask = lane_line_mask(ll)

    return (pred, ll_seg_mask)
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        
        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

        p = Path(p)  # to Path
        #save_path = str(save_dir / p.name)  # img.jpg
        #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            """
            # Write results
            for *xyxy, conf, cls in reversed(det):
                if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if save_img :  # Add bbox to image
                    plot_one_box(xyxy, im0, line_thickness=3)
            """
        # Print time (inference)
        print(f'{s}Done. ({t2 - t1:.3f}s)')
        show_seg_result(im0, (da_seg_mask,ll_seg_mask), is_demo=True)

        """
        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
                print(f" The image with the result is saved in: {save_path}")
            else:  # 'video' or 'stream'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        #w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        #h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        w,h = im0.shape[1], im0.shape[0]
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)
        """
    inf_time.update(t2-t1,img.size(0))
    nms_time.update(t4-t3,img.size(0))
    waste_time.update(tw2-tw1,img.size(0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))
    print(f'Done. ({time.time() - t0:.3f}s)')