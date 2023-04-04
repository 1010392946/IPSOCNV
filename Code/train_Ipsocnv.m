%% claer up
clc
clear
%% Training data prediction data extraction and normalization
%Number of nodes
inputnum=5;
hiddennum=4;
outputnum=4;


data1=load('data\0.2_4x_mat\sim1_4_4100_read_trains.mat');
data2=load('data\0.3_4x_mat\sim1_4_4100_read_trains.mat');
data3=load('data\0.4_4x_mat\sim1_4_4100_read_trains.mat');
data4=load('data\0.2_6x_mat\sim1_6_6100_read_trains.mat');
data5=load('data\0.3_6x_mat\sim1_6_6100_read_trains.mat');
data6=load('data\0.4_6x_mat\sim1_6_6100_read_trains.mat');

data_trains1=load('data\0.2_4x_11.1\sim1_4_4100_trains.txt');
data_trains2=load('data\0.3_4x_11.1\sim1_4_4100_trains.txt');
data_trains3=load('data\0.4_4x_11.1\sim1_4_4100_trains.txt');
data_trains4=load('data\0.2_6x_11.1\sim1_6_6100_trains.txt');
data_trains5=load('data\0.3_6x_11.1\sim1_6_6100_trains.txt');
data_trains6=load('data\0.4_6x_11.1\sim1_6_6100_trains.txt');

% data_trains1 = data1.('sim1_4_4100_read_trains');
% data_trains2 = data2.('sim1_4_4100_read_trains');
% data_trains3 = data3.('sim1_4_4100_read_trains');
% data_trains4 = data4.('sim1_6_6100_read_trains');
% data_trains5 = data5.('sim1_6_6100_read_trains');
% data_trains6 = data6.('sim1_6_6100_read_trains');
data_trains=[data_trains1;data_trains2;data_trains3;data_trains5;data_trains6;];
column=[2,3,4,5,6];
[m1,n1] = size(data_trains);

trainLines = m1;
gdata(1:trainLines,:) = data_trains(1:trainLines,:);

%Random sorting from 1 to trainlines
k=rand(1,trainLines);
[m,n]=sort(k);
%Get input and output data
ginput=gdata(:,column);
goutput1 =gdata(:,7);
%The output changes from one-dimensional to four-dimensional: 0 normal, 1gain, 2hemi_loss, 3homo_loss.
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

%% BP network training
% %Initializing the network structure
net=newff(ginputn,goutput_train,hiddennum);

%Total number of nodes
numsum=inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum;

% Parameter initialization
% Two parameters in the particle swarm algorithm
c1 = 1.49445;
c2 = 1.49445;
lr = 0.5;
history=[];
init_fitness=[]; %Better fit values in the initial archive
init_index=[];%Location in the initial archive
maxgen=13;   % Number of evolutions
sizepop=30;   %Population size
Vmax=1;
Vmin=-1;
popmax=5;
popmin=-5;

%Initialization (large populations)
for i=1:sizepop
    pop(i,:)=5*rands(1,numsum);
    V(i,:)=rands(1,numsum);
    fitness(i)=fun(pop(i,:),inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
end


% Individual and group extremes
[bestfitness bestindex]=min(fitness);
% Putting extreme optima into the archive set
while bestfitness<=100
    init_fitness=[init_fitness bestfitness];
    init_index=[init_index bestindex];
    history=[history bestfitness]; %Record all the optimal fitness values that exist
    pop(bestindex,:)=5*rands(1,numsum);
    V(bestindex,:)=rands(1,numsum);
    fitness(bestindex)=fun(pop(bestindex,:),inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
    [bestfitness bestindex]=min(fitness);
end
%Integrating Archived Sets
init_best=[init_fitness;init_index];
init_pop=pop;%Temporary storage initial pop

zbest=pop(bestindex,:);   %Best Across the Board
gbest=pop;    %Individual Best
fitnessgbest=fitness;   %Individual best-fit values
fitnesszbest=bestfitness;   %Global best fit value
%Initialize the game parameters
strategy_num = 3;
mutation_num = 3; %Variation
flag = zeros(1,strategy_num);%Parameters controlling the variation
p=ones(1,strategy_num)*(1/strategy_num);
success_mem = zeros(1,strategy_num);
failure_mem = zeros(1,strategy_num);
rk = cumsum(ones(1,strategy_num)./strategy_num);
strategy_improve = zeros(1,strategy_num);
%% Iterative Optimization Search
for i=1:maxgen  
    for j=1:sizepop
        probility=rand;
        %Large groups
        if probility<=rk(1)
            strategy = 1;
            V(j,:) = V(j,:)+ (c1+c2)*rand*(gbest(j,:) - pop(j,:));
            V(j,find(V(j,:)>Vmax))=Vmax;
            V(j,find(V(j,:)<Vmin))=Vmin;
        elseif probility<=rk(2)
            strategy = 2;
            V(j,:) = V(j,:)+ (c1+c2)*rand*(zbest - pop(j,:));
            V(j,find(V(j,:)>Vmax))=Vmax;
            V(j,find(V(j,:)<Vmin))=Vmin;
        elseif probility<=rk(3)
            strategy = 3;
            V(j,:) = V(j,:) + c1*rand*(gbest(j,:) - pop(j,:)) + c2*rand*(zbest - pop(j,:));
            V(j,find(V(j,:)>Vmax))=Vmax;
            V(j,find(V(j,:)<Vmin))=Vmin;
        end
        %Control Range
        pop(j,:) = pop(j,:)+V(j,:);
        pop(j,find(pop(j,:)>popmax))=popmax;
        pop(j,find(pop(j,:)<popmin))=popmin;
        %Adaptability value
        fitness(j)=fun(pop(j,:),inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
       
        %Individual optimal update
        if fitness(j) < fitnessgbest(j)
            strategy_improve(strategy)=strategy_improve(strategy)+(fitnessgbest(j)-fitness(j))/fitnessgbest(j);
            gbest(j,:) = pop(j,:); %Record the best position and fitness value
            fitnessgbest(j) = fitness(j);
            success_mem(strategy) = success_mem(strategy) +1;
        else
            failure_mem(strategy) = failure_mem(strategy) +1;
        end
    
        %Population Optimal Update 
        if fitness(j) < fitnesszbest
            zbest = pop(j,:);
            fitnesszbest = fitness(j);
            history=[history fitnesszbest];
        end
    end
    
    %Reproduction Dynamics
    total = success_mem+failure_mem;
    total(find(total == 0)) = 1;
    strategy_improve = strategy_improve./total;
    if isequal(strategy_improve,zeros(1,strategy_num))  %Initialization
       strategy_improve = ones(1,strategy_num);
    end
    strategy_improve(find(strategy_improve == 0)) = 0.1*min(strategy_improve(strategy_improve ~= 0));
    strategy_improve = strategy_improve./sum(strategy_improve);
    f = strategy_improve;
    %p = p+(f-mean(f)).*p;
    p = p+(f-sum(p.*f)).*p.*lr;
    %p = p+(f-sum(p.*f)).*p;
    p(find(p<=0)) = 0;
    p=p./(sum(p)); 
    rk =cumsum(p);
    
    %% Adaptive variation
    for j=1:3
        if flag(1,j)>=mutation_num && p(1,j)>0.5
            flag(1,j)=flag(1,j)+1;
        elseif flag(1,j)<mutation_num && p(1,j)>0.5
            part_improve = p(1,j)-0.5;
            distribe = p;
            distribe(:,j) = 0;
            distribe = part_improve.*(distribe./sum(distribe));
            p = distribe+ p;
            p(1,j)=0.5;%Ensure that the sum of the weights of the three strategies after each variation is 1
            flag(1,j)=flag(1,j)+1;
        end
    end
    %Random point variation
    pos=unidrnd(numsum);
    if rand>0.90
      pop(j,pos)=5*rands(1,1);
    end
    success_mem = zeros(1,strategy_num);
    failure_mem = zeros(1,strategy_num);
    strategy_improve=zeros(1,strategy_num);
end
%Archive Comparison
if isempty(init_best)   
else    
    [init_min index]=min(init_best(1,:));
    if init_min < fitnesszbest
        %zbest = init_best(2,index);
        zbest = init_pop(index,:);
        fitnesszbest = init_min;
    end
end
% %% Analysis of results
% plot(yy)
% title(['适应度曲线  ' '终止代数＝' num2str(maxgen)]);
% xlabel('进化代数');ylabel('适应度');

x=zbest;
figure(1)
f1=plot(history,'rd');
xlabel('time','Fontname','Times New Roman');
ylabel('zbest','Fontname','Times New Roman');
%% Assign the optimal initial threshold weights to the network prediction
% %Value prediction by BP network optimized with genetic algorithm
w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);
B2=B2';

net.iw{1,1}=reshape(w1,hiddennum,inputnum);
net.lw{2,1}=reshape(w2,outputnum,hiddennum);
net.b{1}=reshape(B1,hiddennum,1);
net.b{2}=B2;

%% BP network training
%Network Evolution Parameters
net.trainParam.epochs=200;
net.trainParam.lr=0.1;
net.trainParam.goal=0.00001;

[net,per2]=train(net,ginputn,goutput_train);
save ('PSOBP','net');
