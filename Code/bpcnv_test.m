%%Loading Neural Networks
clc
clear

data_truth1=load('groundtruth_11_1.mat');
data_truth=data_truth1.('data_truth');

% for i=1:length(data_truth(:,1))
%     data_truth(i,1)=data_truth(i,1)+1
% end
% save('groundtruth_11_1','data_truth')
data1=load('data\0.2_4x_mat\sim1_4_4100_read_trains.mat');
data2=load('data\0.2_6x_mat\sim1_6_6100_read_trains.mat');
data3=load('data\0.3_4x_mat\sim1_4_4100_read_trains.mat');
data4=load('data\0.3_6x_mat\sim1_6_6100_read_trains.mat');
data5=load('data\0.4_4x_mat\sim1_4_4100_read_trains.mat');
data6=load('data\0.4_6x_mat\sim1_6_6100_read_trains.mat');
% 
data_trains1=load('data\0.2_4x_11.1\sim1_4_4100_trains.txt');
data_trains2=load('data\0.3_4x_11.1\sim1_4_4100_trains.txt');
data_trains3=load('data\0.4_4x_11.1\sim1_4_4100_trains.txt');
data_trains4=load('data\0.2_6x_11.1\sim1_6_6100_trains.txt');
data_trains5=load('data\0.3_6x_11.1\sim1_6_6100_trains.txt');
data_trains6=load('data\0.4_6x_11.1\sim1_6_6100_trains.txt');

% data_trains1 = data1.('sim1_4_4100_read_trains');
% data_trains2 = data2.('sim1_6_6100_read_trains');
% data_trains3 = data3.('sim1_4_4100_read_trains');
% data_trains4 = data4.('sim1_6_6100_read_trains');
% data_trains5 = data5.('sim1_4_4100_read_trains');
% data_trains6 = data6.('sim1_6_6100_read_trains');
data_trains=[data_trains1;data_trains2;data_trains3;data_trains4;data_trains5;data_trains6];

column=[2,3,4,5,6];
[m1,n1] = size(data_trains);
[m3,n3]=size(data_truth);
truthLines=m3;
trainLines = m1;
gdata(1:trainLines,:) = data_trains(1:trainLines,:);
gdatatruth(1:truthLines,:)=data_truth(1:truthLines,:);
column3=[1,2];
% gdatatruth_bin=gdatatruth(:,column3);
groundtruth=data_truth(:,column3); 
gtRev=fliplr(groundtruth(:,2)'); 
% gdatatruth_bin=gdatatruth_bin(:);
% gdatatruth_bin=gdatatruth_bin';
%Random sorting from 1 to trainlines
k=rand(1,trainLines);
[m,n]=sort(k);
%Get input and output data
ginput=gdata(:,column);
%goutput1 =gdata(:,6);
goutput1 =gdata(:,7);
%The output changes from one dimension to four dimensions
goutput=zeros(trainLines,4);
for i=1:trainLines
    switch goutput1(i)
        case 0
            goutput(i,:)=[1 0 0 0];
        case 1
            goutput(i,:)=[0 1 0 0];
        case 2
            goutput(i,:)=[0 0 1 0];
	case 3
            goutput(i,:)=[0 0 0 1];
    end
end
%Find the training data and prediction data

ginput_train=ginput(n(1:trainLines),:)';
goutput_train=goutput(n(1:trainLines),:)';


%Sample input and output data normalization
[ginputn,ginputps]=mapminmax(ginput_train);
[outputn,outputps]=mapminmax(goutput_train);

%%Loading Network
load('-mat','PSOBP');
TPR=zeros(6,50);
FPR=zeros(6,50);
boundary=zeros(6,50); %Boundaries for each bam, used to generate box plots
for l=0.2:0.1:0.4
    for m=4:2:6
        %Numbering for calculation
        if l<=0.25
            if m==4
                label=1;
            else
                label=2;
            end
        elseif l<=0.35
            if m==4
                label=3;
            else
                label=4;
            end
        elseif l<=0.45
            if m==4
                label=5;
            else
                label=6;
            end
        end
        %column=[2,3,4,5];
        num=50;
        TP_count_sum=0;
        TPFP_count_sum=0;

        bound = 0;
        count_bias_sum=0;
        %boundary=[]; %Boundaries for each bam, used to generate box plots
        num_boundary=0;



        for t=1:num
            %data2=load(['data\',num2str(l),'_',num2str(m),'x_mat\sim',num2str(t) ,'_',num2str(m),'_',num2str(m),'100_read_trains.mat']);
            data_tests=load(['data\',num2str(l),'_',num2str(m),'x_11.1\sim',num2str(t) ,'_',num2str(m),'_',num2str(m),'100_trains.txt']);
            %data_tests = data2.(['sim', num2str(t) ,'_',num2str(m),'_',num2str(m),'100_read_trains']);
            [m2,n2] = size(data_tests);
            testLines=m2;
            gdata2(1:testLines,:) = data_tests(1:m2,:);
            ginput2_bin=gdata2(:,1);
            ginput2=gdata2(:,column);
            %goutput1 =gdata2(:,6);
            goutput1 =gdata2(:,7);
            goutput2=zeros(testLines,4);
            for i=1:testLines
                switch goutput1(i)
                    case 0
                        goutput2(i,:)=[1 0 0 0];
                    case 1
                        goutput2(i,:)=[0 1 0 0];
                    case 2
                        goutput2(i,:)=[0 0 1 0];
                    case 3
                        goutput2(i,:)=[0 0 0 1];
                end
            end
            ginput_test=ginput2((1:testLines),:)';
            goutput_test=goutput2((1:testLines),:)';
    %% Projections
            %Predictive data normalization
            inputn_test=mapminmax('apply',ginput_test,ginputps);

            %Network Prediction Output
            an=sim(net,inputn_test);

            %Network output inverse normalization
            BPoutput=mapminmax('reverse',an,outputps);

            %Prediction error
            error=BPoutput-goutput_test;
            abs_error=abs(error);
            errorsum=sum(abs(error));

             
            fid=fopen([num2str(l),'_',num2str(m),'x_binnumber\sim', num2str(t) ,'_',num2str(m),'_',num2str(m),'100_bin_number.txt'],'wt'); %Write the bin_number array to the file
            P_count=0;
            TPFP_count=0;
            k=1;
            kkk=1;
            binnumber=[];
            binnumber_tpfp=[];
            for q=1:testLines
                if (( abs_error(2,q) < abs_error(1,q) && abs_error(2,q) < abs_error(3,q) && abs_error(2,q) < abs_error(4,q) && goutput_test(2,q) == 1) || ( abs_error(3,q) < abs_error(1,q) && abs_error(3,q) < abs_error(2,q) && abs_error(3,q) < abs_error(4,q) && goutput_test(3,q) == 1) || (abs_error(4,q) < abs_error(1,q) && abs_error(4,q) < abs_error(2,q) && abs_error(4,q) < abs_error(3,q) && goutput_test(4,q) == 1)) 
                    TP_count=TP_count+1;
                    binnumber(k)=ginput2_bin(q);
                    k=k+1;
                end
                if ( goutput_test(2,q) == 1 || goutput_test(3,q) == 1 || goutput_test(4,q) == 1 )
                    P_count=P_count+1;
                    binnumber_tpfp(kkk)=ginput2_bin(q);
                    kkk=kkk+1;
                end
                if (  (abs_error(2,q) < abs_error(1,q) && abs_error(2,q) < abs_error(3,q) && abs_error(2,q) < abs_error(4,q) ) || ( abs_error(3,q) < abs_error(1,q) && abs_error(3,q) < abs_error(2,q) && abs_error(3,q) < abs_error(4,q)) || (abs_error(4,q) < abs_error(1,q) && abs_error(4,q) < abs_error(2,q) && abs_error(4,q) < abs_error(3,q)) )
                    TPFP_count=TPFP_count+1;
                end
            end
            fclose(fid);
    
            % Calculation of TPR
            TPR(label,t)=TP_count./TPFP_count;
            % Calculation of FPR
            %FPR(t)=(TPFP_count-TP_count)./(1512*14);
            FPR(label,t)=(TPFP_count-TP_count)./(testLines-P_count);

            TPFP_count_sum=TPFP_count_sum+TPFP_count;
            TP_count_sum=TP_count_sum+TP_count;
            boundbias = (P_count - TP_count)./14;
            bound = bound + boundbias;
    
            % The detected 14 cnv's bin beginning and ending number is saved
            binnumberRev=fliplr(binnumber);
            [m3,n3]=size(binnumber);
            jj=1;
            bin1=[];
            for ii=1:n3
                if binnumber(ii) >= groundtruth(jj,1)
                    bin1(jj)=binnumber(ii);
                    jj=jj+1;
                end
                if jj>14
                    break; 
                end
            end
            jjj=1;
            bin22=[];
            for iii=1:n3
                if binnumberRev(iii) <= gtRev(jjj)
                    bin22(jjj)=binnumberRev(iii);
                    jjj=jjj+1;
                end
                if jjj>14
                    break; 
                end
            end
            bin2=fliplr(bin22);
            [m5,n5]=size(bin1);
            [m6,n6]=size(bin2);
            if(n5<14 || n6<14)
                continue; 
            end
            bin=[bin1;bin2]'; 
            % Calculation of boundary accuracy
            [m4,n4]=size(bin);
            c_bias1=[];
            c_bias2=[];
            for rr=1:m4
                c_bias1(rr)=bin(rr,1)-groundtruth(rr,1);
                c_bias2(rr)=groundtruth(rr,2)-bin(rr,2);
            end
            c_bias=sum(c_bias1)+sum(c_bias2);
            count_bias=c_bias./14;
            if count_bias > 50
              continue;
            end
            boundary(label,t)=count_bias;
            num_boundary = num_boundary + 1;
            count_bias_sum = count_bias_sum+count_bias;
            binnumber_fp=setdiff(binnumber_tpfp,binnumber);
    
        end
       %% Analysis of results 

        TP_count_avg=TP_count_sum/num;
        TPFP_count_avg=TPFP_count_sum/num;
        recall=TP_count_avg/P_count;
        precision=TP_count_avg/TPFP_count_avg;
        F1_score=(2*recall*precision)/(recall+precision);
        boundary_bias=bound/num;
        count_bias_avg=count_bias_sum/num_boundary;
        disp([num2str(l),'_',num2str(m),'x-sensitivity:']);
        disp(recall);
        disp([num2str(l),'_',num2str(m),'x-precision:']);
        disp(precision);
        disp([num2str(l),'_',num2str(m),'x-F1-score:']);
        disp(F1_score);
%disp([num2str(l),'_',num2str(m),'x-all_bound_bias:']);
%disp(boundary_bias);
%disp('14_bound_bias:');
%disp(count_bias_avg);
%disp('bound_bias:');
%disp(boundary);
        %disp([num2str(l),'_',num2str(m),'x-TPR:']);
        %disp(TPR);
        %disp([num2str(l),'_',num2str(m),'x-FPR:']);
        %disp(FPR);
    end
end