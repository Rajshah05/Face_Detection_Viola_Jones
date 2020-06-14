import numpy as np
import os
import sys
import pickle
import cv2
import json

global cascaded

def varnorm(img):
    img = np.asarray(img)
    # mean = np.sum(np.sum(img))/576
    mean = np.mean(img)
    # std = np.sqrt(mean**2 - np.sum(np.sum(np.multiply(img,img)))/576)
    std = np.std(img)
    # print("std",std)
    img = ((img-mean)/float(std))

    return img

file2 = open("Model_Files/cascade.pkl", "rb")
cascaded = pickle.load(file2)

def det_face(integimg,coor):
    
    global cascaded
    f_score = np.zeros(len(integimg))
    
    for jj, vv in enumerate(cascaded):
        stage_v = np.zeros(len(integimg))
        is_face = np.zeros(len(integimg))
        for j,v in enumerate(vv[0]):
            if v[2][2] == 0:
                ir = v[2][0][0]
                ic = v[2][0][1]
                iir = v[2][1][0]
                iic = v[2][1][1]
                if len(integimg)==1:
                    fv = integimg[ir-1,ic-1] - integimg[ir-1,iic] + 2*integimg[ir+((iir-ir)//2),iic] - integimg[iir,iic] + integimg[iir,ic-1] - 2*integimg[ir+((iir-ir)//2),ic-1]    
                else:
                    fv = integimg[:,ir-1,ic-1] - integimg[:,ir-1,iic] + 2*integimg[:,ir+((iir-ir)//2),iic] - integimg[:,iir,iic] + integimg[:,iir,ic-1] - 2*integimg[:,ir+((iir-ir)//2),ic-1]
            elif v[2][2] == 1:
                ir = v[2][0][0]
                ic = v[2][0][1]
                iir = v[2][1][0]
                iic = v[2][1][1]
                if len(integimg)==1:
                    fv = -integimg[ir-1,ic-1] + 2*integimg[ir-1,ic+((iic-ic)//2)] - integimg[ir-1,iic] + integimg[iir,iic] - 2*integimg[iir,ic+((iic-ic)//2)] + integimg[iir,ic-1]
                else:
                    fv = -integimg[:,ir-1,ic-1] + 2*integimg[:,ir-1,ic+((iic-ic)//2)] - integimg[:,ir-1,iic] + integimg[:,iir,iic] - 2*integimg[:,iir,ic+((iic-ic)//2)] + integimg[:,iir,ic-1]
            elif v[2][2] == 2:
                ir = v[2][0][0]
                ic = v[2][0][1]
                iir = v[2][1][0]
                iic = v[2][1][1]
                if len(integimg)==1:
                    fv = -integimg[ir-1,ic-1] + 2*integimg[(ir+(iir-ir)//3),ic-1] - 2*integimg[(ir+2*((iir-ir)//3)+1),ic-1] + integimg[iir,ic-1] + integimg[ir-1,iic] - 2*integimg[(ir+(iir-ir)//3),iic] + 2*integimg[(ir+2*((iir-ir)//3)+1),iic] - integimg[iir,iic]
                else:
                    fv = -integimg[:,ir-1,ic-1] + 2*integimg[:,(ir+(iir-ir)//3),ic-1] - 2*integimg[:,(ir+2*((iir-ir)//3)+1),ic-1] + integimg[:,iir,ic-1] + integimg[:,ir-1,iic] - 2*integimg[:,(ir+(iir-ir)//3),iic] + 2*integimg[:,(ir+2*((iir-ir)//3)+1),iic] - integimg[:,iir,iic]
            elif v[2][2] == 3:
                ir = v[2][0][0]
                ic = v[2][0][1]
                iir = v[2][1][0]
                iic = v[2][1][1]
                if len(integimg) ==1:
                    fv = -integimg[ir-1,ic-1] + 2*integimg[ir-1,(ic+((iic-ic)//3))] - 2*integimg[ir-1,ic+(2*(iic-ic)//3)+1] + integimg[ir-1,iic] + integimg[iir,ic-1] - 2*integimg[iir,(ic+((iic-ic)//3))] + 2*integimg[iir,ic+(2*((iic-ic)//3))+1] - integimg[iir,iic]
                else:
                    fv = -integimg[:,ir-1,ic-1] + 2*integimg[:,ir-1,(ic+((iic-ic)//3))] - 2*integimg[:,ir-1,ic+(2*(iic-ic)//3)+1] + integimg[:,ir-1,iic] + integimg[:,iir,ic-1] - 2*integimg[:,iir,(ic+((iic-ic)//3))] + 2*integimg[:,iir,ic+(2*((iic-ic)//3))+1] - integimg[:,iir,iic]
            elif v[2][2] == 4:
                ir = v[2][0][0]
                ic = v[2][0][1]
                iir = v[2][1][0]
                iic = v[2][1][1]
                if len(integimg)==1:
                    fv = -integimg[ir-1,ic-1] + 2*integimg[ir-1,ic+((iic-ic)//2)] - integimg[ir-1,iic] + 2*integimg[ir+((iir-ir)//2),ic-1] - 4*integimg[ir+((iir-ir)//2),ic+((iic-ic)//2)] +2*integimg[ir+((iir-ir)//2),iic] - integimg[iir,ic-1] + 2* integimg[iir,ic+((iic-ic)//2)] - integimg[iir,iic]    
                else:
                    fv = -integimg[:,ir-1,ic-1] + 2*integimg[:,ir-1,ic+((iic-ic)//2)] - integimg[:,ir-1,iic] + 2*integimg[:,ir+((iir-ir)//2),ic-1] - 4*integimg[:,ir+((iir-ir)//2),ic+((iic-ic)//2)] +2*integimg[:,ir+((iir-ir)//2),iic] - integimg[:,iir,ic-1] + 2* integimg[:,iir,ic+((iic-ic)//2)] - integimg[:,iir,iic]
            ffv = np.zeros(len(integimg))
            ffv[v[3]*fv > v[3]*v[4]*np.ones(len(integimg))] = 1
#             if v[3]*fv > v[3]*v[4]:
#                 ffv = 1
#             else:
#                 ffv = 0
#             alpha = np.log(1/v[1])
            alphas = np.ones(len(integimg))*np.log(1/v[1])
            
            stage_v = stage_v + np.multiply(alphas,ffv)
        th = np.ones(len(integimg))*vv[1]
        is_face[stage_v>th] = 1
#         print(stage_v.shape, th.shape, f_score.shape)
        f_score = f_score + (stage_v - th)
#         print(len(integimg), len(is_face))
        integimg = integimg[is_face==1]
        f_score = f_score[is_face==1]
        coor = coor[is_face==1]
#         print(len(integimg))
    return (integimg,f_score,coor)




folder = sys.argv[1]
# folder = "test_images"
rs = 4
final_jason = []
count = 0
for filename in os.listdir(folder):
    count += 1
    print("count", count)
    print(filename)
    img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
    
    img = cv2.resize(img, (0,0), fx=1/rs, fy=1/rs)
    num_r = img.shape[0]
    num_c = img.shape[1]
    # pr_fd_thr = 0
    # cr_fd_thr = 0
    scale = 5
    # scale = 1

    f_score_l = []
    coor_list = []
    swl = []
    while scale < 130:
        sub_win_len = int(scale*24/rs)
        print(scale)

        sub_win_arr = []
        coor = []
        for ir in range(num_r-(sub_win_len-1)):
            for ic in range(num_c-(sub_win_len-1)):
                sub_win = img[ir:ir+sub_win_len,ic:ic+sub_win_len]
                sub_win = cv2.resize(sub_win, (24,24))
                sub_win = varnorm(sub_win)
                sub_win = cv2.integral(sub_win)
                sub_win_arr.append(sub_win)
                coor.append((ir,ic))


        sub_win_arr = np.asarray(sub_win_arr)
    #     print(sub_win_arr.shape)
        coor = np.asarray(coor)
        if len(sub_win_arr) < 1:
            break
        (face_sw, f_score, coor) = det_face(sub_win_arr,coor)
        print(coor.shape,"coor shape")
        print(f_score.shape,"f_score shape")
        f_score_l += list(f_score)
        coor_list += list(coor)
        swl += [sub_win_len]*len(coor)
        scale = scale*5
        
        
        f_score_list = np.asarray(f_score_l)
        max_f_score = np.max(f_score_list)
        max_f_score_ind = np.argmax(f_score_list) 
#         print(max_f_score_ind)
#         print(max_f_score)
        T = 0.965
        final_faces = []
        for ii,vv in enumerate(f_score_list):
            if vv/max_f_score > T:
                final_faces.append(ii)
        
        
        for hh in final_faces:
            final_jason.append({"iname": filename, "bbox": [int(coor_list[hh][0]*rs), int(coor_list[hh][1]*rs), int(swl[hh]*rs), int(swl[hh]*rs)]})
#             cv2.rectangle(img0,(coor_list[hh][1]*rs,coor_list[hh][0]*rs),(coor_list[hh][1]*rs+swl[hh]*rs,coor_list[hh][0]*rs+swl[hh]*rs),(255, 0, 0),1)
output_json = "results.json"
#dump json_list to result.json
with open(output_json, 'w') as f:
    json.dump(final_jason, f)
#         cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
#         cv2.imshow('image', img0)
#         cv2.waitKey(5000)
#         cv2.destroyAllWindows()
            