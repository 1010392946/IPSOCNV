import numpy as np
import pysam
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
import re
import math
# extract feature values by loci, select the first duplicate loci, binsize=1000

def get_chrlist(filename):
    # Read the chromosome sequence from the bam file to see how many chromosomes are in the bam file, in the simulation data there is only chromosome 21
    # Input: bam Output: list of chromosomes
    samfile = pysam.AlignmentFile(filename, "rb")
    List = samfile.references
    chrList = np.full(len(List), 0)
    for i in range(len(List)):
        chr = str(List[i]).strip('chr')
        if chr.isdigit():
            chrList[i] = int(chr)

    return chrList



def get_RC(filename, chrList, ReadCount):
    # Extract the read count value of each locus from the bam file
    # Input: bam file, list of chromosomes and initialized rc array Output: rc array
    samfile = pysam.AlignmentFile(filename, "rb")
    for line in samfile:
        if line.reference_name:
            chr = line.reference_name.strip('chr')
            if chr.isdigit():
                num = np.argwhere(chrList == int(chr))[0][0]
                posList = line.positions
                ReadCount[num][posList] += 1

    return ReadCount


def read_ref_file(filename, ref, num):
    # Read reference file
    # Input: fasta file, initialized ref array and current chromosome number Output: ref array
    if os.path.exists(filename):
        print("Read reference file: " + str(filename))
        with open(filename, 'r') as f:
            line = f.readline()
            for line in f:
                linestr = line.strip()
                ref[num] += linestr
    else:
        print("Warning: can not open " + str(filename) + '\n')
    return ref


def ReadDepth(ReadCount, binNum, ref):
    # Get the read depth value of each bin from the rc array

    RD = np.full(binNum, 0.0)
    GC = np.full(binNum, 0)
    pos = np.arange(1, binNum+1)
    for i in range(binNum):
        RD[i] = np.mean(ReadCount[i*binSize:(i+1)*binSize])
        cur_ref = ref[i*binSize:(i+1)*binSize]
        N_count = cur_ref.count('N') + cur_ref.count('n')
        if N_count == 0:
            gc_count = cur_ref.count('C') + cur_ref.count('c') + cur_ref.count('G') + cur_ref.count('g')
        else:
            RD[i] = -10000
            gc_count = 0
        GC[i] = int(round(gc_count / binSize, 3) * 1000)

    index = RD > 0
    RD = RD[index]
    GC = GC[index]
    pos = pos[index]
    #RD = gc_correct(RD, GC)

    return pos, RD , GC


for z in range(1,51):
#for z in range(1,51):
    # get params
    num = z;

    bam = "/data/0.4-6x/sim" + str(z) + "_6_6100_read.sort.bam";

    binSize = 1000
    chrList = get_chrlist(bam)

    chrNum = len(chrList)
    refList = [[] for i in range(chrNum)]
    for i in range(chrNum):
        reference = "/home/data/chr21.fa"
        refList = read_ref_file(reference, refList, i)

    chrLen = np.full(chrNum, 0)
    for i in range(chrNum):
        chrLen[i] = len(refList[i])
    print("Read bam file:", bam)
    ReadCount = np.full((chrNum, np.max(chrLen)), 0)
    ReadCount = get_RC(bam, chrList, ReadCount)
    for i in range(chrNum):
        binNum = int(chrLen[i]/binSize)+1
        pos, RD, GC = ReadDepth(ReadCount[0], binNum, refList[i])
#        plot(pos, RD)

    #==========================================================step2. GC normalization
    GC=np.array(GC,dtype='float32').reshape(1,-1)
    normal_GC = Normalizer(norm='max').fit_transform(GC)
    normal_GC = normal_GC.flatten()


    #========================================================== step3. input GroundTruthCNV
    myin = open('/home/data/GroundTruthCNV.txt','r')
    #next(myin)
    line = myin.readline()
    list1 = []
    while line :
        a = line.split()
        b = a[0:2]
        list1.append(b)
        line = myin.readline()
    myin.close()

    list2 = list1[1:]
    #for i in list2:
    #   print (i)

    #Convert bitpoint string arrays to bitpoint value arrays
    list2 = np.array(list2,dtype=int)
    #print(list2)

    #Convert the bit array into a bin array, view the bin corresponding to the bit
    #list3 = list2 /binSize
    for i in range(list2.shape[0]):
        for j in range(0,2):
            list2[i][j]= math.ceil(list2[i][j] / binSize)
    list3 = np.array(list2,dtype=int)
    #print(list3)

    #Relevance array definition
    avg1 = np.full(5, 0.0)
    avg1 = avg1.astype(np.float64)
    avg2 = np.full(binNum-5, 0.0)
    avg2 = avg2.astype(np.float64)
    avg3 = np.full(5, 0.0)
    avg3 = avg1.astype(np.float64)


    samfile = pysam.AlignmentFile("/home/data/0.4-6x/sim" + str(z) +"_6_6100_read.sort.bam")
    map_q = []
    pos_q = []
    read_num_list = []
    read_num = np.full(binNum, 0.0)
    pos_read = np.full(len(pos), 0.0)
    t=0
    for r in samfile:
        map_q.append(r.mapq)
        pos_q.append(r.pos)
        pos_line = r.positions
        if len(pos_line) == 0:
            continue
        bin_line_left = math.ceil(pos_line[0] / binSize)
        bin_line_right = math.ceil(pos_line[-1] / binSize)
        for i in range(bin_line_left,bin_line_right+1):
            if i in pos:
                read_num[i] = read_num[i] + 1
                if i in pos_read:
                    continue
                else:
                    pos_read[t] = i
                    t += 1
            else:
                read_num[i] = 0.0
    for i in range(len(read_num)):
        if read_num[i]!=0.0:
            read_num_list.append(read_num[i])
    #read_num_list = list(read_num)
    #read_num_list.remove(0)
    pos_read.sort()
    read_num = np.array(read_num_list,dtype = 'float32').reshape(1,-1)
    normal_read_num = Normalizer(norm='max').fit_transform(read_num)
    normal_read_num = normal_read_num.flatten()
    #read_new = list(zip(pos_read, normal_read_num))
    posmapq_old = list(zip(pos_q, map_q))
    # Calculate the average quality information of duplicate loci into a two-dimensional array posmapq_new

    pos_q_new = [0]
    map_q_new = [0]
    pos_q_new[0] = pos_q[0]
    map_q_new[0] = map_q[0]
    j = 0
    k = 0
    for i in pos_q:
        if i != pos_q_new[j]:
            pos_q_new.append(pos_q[k])
            map_q_new.append(map_q[k])
            j += 1
        k += 1
    pos_q_new = np.array(pos_q_new,dtype=int)
    pos_bin = pos_q_new / binSize
    pos_bin = np.array(pos_bin,dtype=int)
    #posmapq_new = list(zip(pos_q_new,map_q_new))
    print("begin to bin")

    #Calculate the average mass of each bin and store it in the binmapq array
    bin_mapq = np.full(len(pos),0.0)
    bin_mapq = bin_mapq.astype(np.float64)
    sum_q = np.full(len(pos),0.0)
    sum_q = sum_q.astype(np.float64)
    count = np.full(len(pos),0)
    count = count.astype(np.int_)


    pos_bin_new_num=0
    map_q_new_2_num=0

    for i in range(len(pos_bin)):
        if pos_bin[i] == pos[0]:
            #pos_bin_new_num = len(pos_bin) - i
            #map_q_new_2_num = len(pos_bin) - i
            pos_bin_new = np.full(len(pos_bin) - i, 0);
            map_q_new_2 = np.full(len(pos_bin) - i, 0);
            x = i
            break
    #pos_bin_new = np.full(pos_bin_new_num, 0)
    #map_q_new_2 = np.full(map_q_new_2_num, 0)
    sum_q_2=np.full(len(pos),0)
    count_2=np.full(len(pos),0)
    bin_mapq_2=np.full(len(pos),0.0)
    for i in range(len(pos_bin_new)):
        pos_bin_new[i] = pos_bin[x]
        map_q_new_2[i] = map_q_new[x]
        x+=1
    m = 0
    pos_bin_list = pos_bin.tolist()
    for i in range(len(pos_bin_new)):
            if (pos_bin_new[i] == pos[m]) :
                sum_q_2[m] += map_q_new_2[i]
                count_2[m] += 1
            elif(pos_bin_new[i] > pos[m]):
                sum_q_2[m+1] += map_q_new_2[i]
                count_2[m+1] += 1
                m += 1
            elif(pos_bin_new[i] < pos[m]):
                continue;
    for n in range(len(sum_q)):
        if count_2[n] == 0:
            count_2[n] = 100
        bin_mapq_2[n] = sum_q_2[n]/count_2[n]
        if bin_mapq_2[n] == 0:
            bin_mapq_2[n] = 60
    #====================================================Compare quality
    bin_mapq_2 = np.array(bin_mapq_2, dtype='float32').reshape(1, -1)
    normal_bin_mapq_2 = Normalizer(norm='max').fit_transform(bin_mapq_2)
    normal_bin_mapq_2 = normal_bin_mapq_2.flatten()

    print("begin output to files")

    # ========================================================== step4 output to files, with trains
    myOut2 = open('/home/data/simdata/0.4-6x/sim' + str(z) + '_6_6100_trains.txt', 'w')
    #myOut2 = open('/home/wangxuan/data/simdata/30x/sim' + str(z) + '_0.8_30x_trains.txt','w')
    #myOut.write("bin" + "\t" + "rd" + "\t" + "gc" + "\t" + "rel" + "\t" + "qua" + "\t" + "label" + "\n")
    for i in range(len(pos)):
        myOut2.write(str(pos[i]))      #bin
        myOut2.write("\t"+str(RD[i]))  #rd
        myOut2.write("\t"+str(normal_GC[i]*10))  #gc
        #Relevance
        if (pos[i] - pos[0]) < 5 :
            avg1[i] = RD[i] - (sum(RD[(i + 1):(i + 5)]) / 5)
            myOut2.write("\t"+str(abs(avg1[i])))
        elif ((pos[len(pos)-1] - pos[len(pos)-1-i])) < 5:
            avg3[i] = RD[i] - (sum(RD[(i - 5):(i - 1)]) / 5)
            myOut2.write("\t"+str(abs(avg3[i])))
        else:
            avg2[i] = ((RD[i] - (sum(RD[(i - 5):(i - 1)]) / 5)) + (RD[i] - (sum(RD[(i + 1):(i + 5)]) / 5))) / 2
            myOut2.write("\t" + str(abs(avg2[i])))

        myOut2.write("\t" + str(normal_bin_mapq_2[i]))
        if ((pos == pos_read).all()):
            myOut2.write("\t" + str(normal_read_num[i]))
        # label     0:normal；1：gain；2：hemi_loss;3:homo_loss;
        for j in range(list3.shape[0]):
            if (list3[j][0] <= pos[i] <= list3[j][1] and j <= 5):
                myOut2.write("\t" + "1" + "\n")
                break
            elif (list3[j][0] <= pos[i] <= list3[j][1] and 5 < j <= 9):
                myOut2.write("\t" + "2" + "\n")
                break
            elif (list3[j][0] <= pos[i] <= list3[j][1] and 9 < j <= 13):
                myOut2.write("\t" + "3" + "\n")
                break
            else:
                if j == (list3.shape[0] - 1):
                    myOut2.write("\t" + "0" + "\n")
    print("length:" + str(pos.shape[0]))
    myOut2.close()

