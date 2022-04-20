%Trabalho final do mestrado, usando a base de dados
%Versao onde tem o for dentro do for para fazer todas as bases

%% Inicializacao
clear variables
close all
clc

rng default

%% Dados


xls = '.xlsx';
path = 'data/UG01/';
        

baseInicial =1;
baseFinal = 9;
bases = (baseFinal-baseInicial)+1;


folds = 5; %valor entre 5 e 10;

MSE_MLP = zeros(folds,bases);
MSE_SVM = zeros(folds,bases);
MSE_LM = zeros(folds,bases);
MSE_RB = zeros(folds,bases);
MSE_GRNN = zeros(folds,bases);

R_MLP = zeros(folds,bases);
R_SVM = zeros(folds,bases);
R_LM = zeros(folds,bases);
R_RB = zeros(folds,bases);
R_GRNN = zeros(folds,bases);

STD_MLP = zeros(folds,bases);
STD_SVM = zeros(folds,bases);
STD_LM = zeros(folds,bases);
STD_RB = zeros(folds,bases);
STD_GRNN = zeros(folds,bases);



for j=baseInicial:baseFinal

a = j;    
a = num2str(a); 
    
filename = strcat(path,a,xls);

[data, ~, ~] = xlsread(filename);
clear a;

data = rmoutliers(data, 'grubbs');
Maximo_variaveis = 40;
Maximo_variaveis = Maximo_variaveis -1;

X = [data(2:end, 1:13) data(2:end, 15:end)];
Y = data(2:end, 14); % vibração -> Alvo

epocas = 1000;

clear data

%% Normalização [-1, 1]
Xn = mapminmax(X')';
Yn = mapminmax(Y')';

clear X Y

%Fazer uma selecao de parametros baseado em correlacao linear.

%Selecao de parametros

[r] = corrcoef(Xn); %Retorna a correlacao linear das minhas variaveis
%r = matriz de correcao, quanto maior mais correlacionado um ponto com o
%outro

%Selecionando os dados baseado na correlacao linar de Pearson
[~,rindex] = sort(abs(r(1,:)),'ascend');
Xmenor = Xn(:,[1 rindex(1:Maximo_variaveis)]);

%FAzendo o K-fold
Validacao_cruzada = cvpartition(length(Yn),'KFold', folds); %estrutura da validacao  cruzada


neuronios_CamadaEscondida = 4;


netMLP = cell(folds,bases);
SVM = cell(folds,bases); %Mdl — Trained SVM regression model
LM = cell(folds,bases); %LM — Trained Linear Model


    
for i=1:folds
   
    
    idxTreino = training(Validacao_cruzada,i);
    xTreino = Xmenor(idxTreino,:)';
    yTreino = Yn(idxTreino,:)';

    %inicializada da rede MLP
    netMLP{i,j} = feedforwardnet(neuronios_CamadaEscondida);
    netMLP{i,j} = configure(netMLP{i,j},xTreino,yTreino);
    netMLP{i,j}.divideParam.trainRatio = 1;
    netMLP{i,j}.divideParam.valRatio = 0;    
    netMLP{i,j}.divideParam.testRatio = 0;
    netMLP{i,j}.trainFcn = 'trainscg';
    % Funções de Treinamento: 
    % traingd:  Backpropagation
    % traingdm: Backpropagation with Momentum    
    % traingdx: Variable Learning Rate Backpropagation
    % trainlm:  Levenberg-Marquardt
    % trainbfg: BFGS Quasi-Newton
    % trainrp:  Resilient Backpropagation
    % trainscg: Scaled Conjugate Gradient
    % traincgb: Conjugate Gradient with Powell/Beale Restarts
    % traincgf: Fletcher-Powell Conjugate Gradient
    % traincgp: Polak-Ribiére Conjugate Gradient
    % trainoss: 
    
    netMLP{i,j}.performFcn = 'mse';
    netMLP{i,j}.trainParam.epochs = epocas;
    netMLP{i,j}.trainParam.showWindow = false;
    netMLP{i,j}.layers{1:end-1}.transferFcn = 'tansig';
    netMLP{i,j}.layers{end}.transferFcn = 'purelin';   
    netMLP{i,j} = init(netMLP{i,j});
    netMLP{i,j} = train(netMLP{i,j},xTreino,yTreino);
    
  
    idxTeste = test(Validacao_cruzada,i);
    xTeste = Xmenor(idxTeste,:)';
    yTeste = Yn(idxTeste,:)';
    
    ySimulado_MLP = sim(netMLP{i,j},xTeste);
    MSE_MLP(i,j) = mse(netMLP{i,j},yTeste,ySimulado_MLP);
    
    r = corrcoef(yTeste,ySimulado_MLP);
    R_MLP(i,j) = r(1,2);
    clear r;
    
    STD_MLP(i,j) = std(ySimulado_MLP - yTeste);
    
  %inicializada da rede Base Radial
     netRB{i,j} = newrb(xTreino,yTreino,0,1,2);
     ySimulado_RB = sim(netRB{i,j},xTeste);    
     MSE_RB(i,j) = sqrt(mse(netRB{i,j},yTeste,ySimulado_RB));
    
    r = corrcoef(yTeste,ySimulado_RB);
    R_RB(i,j) = r(1,2);
    clear r;
  
        STD_RB(i,j) = std(ySimulado_RB - yTeste);
    
    %inicializada da rede Base Radial Completa
     netGRNN{i,j} = newgrnn(xTreino,yTreino);
     ySimulado_GRNN = sim(netGRNN{i,j},xTeste);    
     MSE_GRNN(i,j) = sqrt(mse(netRB{i,j},yTeste,ySimulado_GRNN));
  
    r = corrcoef(yTeste,ySimulado_GRNN);
    R_GRNN(i,j) = r(1,2);
    clear r;
  
        STD_GRNN(i,j) = std(ySimulado_GRNN - yTeste);
    
 %*********************************   
  %MVS / SVM  
     SVM{i,j} = fitrsvm(xTreino',yTreino'); %Máquina de Vetores Suporte
     ySimulado_SVM = predict(SVM{i,j},xTeste');
     ISE_SVM = ((ySimulado_SVM - yTeste').^2); 
     
%Performance do SVM     
for k = 1:length(ISE_SVM)
    RMSE2(k) = sqrt(mean(ISE_SVM(1:k)));
end
MSE_SVM(i,j) = mean(RMSE2);
 
    r = corrcoef(yTeste,ySimulado_SVM);
    R_SVM(i,j) = r(1,2);
    clear r;

         STD_SVM(i,j) = std(ySimulado_SVM - yTeste');
    
%*********************************  


%*********************************  
  %LM - Fit linear regression model
     LM{i} = fitlm(xTreino',yTreino'); %LM
      ySimulado_LM = predict(LM{i},xTeste');
     ISE_LM = ((ySimulado_LM - yTeste').^2); 
     
%Performance do LM     
for k = 1:length(ISE_LM)
    RMSE3(k) = sqrt(mean(ISE_LM(1:k)));
end
MSE_LM(i,j) = mean(RMSE3);

    r = corrcoef(yTeste,ySimulado_LM);
    R_LM(i,j) = r(1,2);
    clear r;
     STD_LM(i,j) = std(ySimulado_LM - yTeste'); 
    

end

% RMSE_Medio_MLP = mean(MSE_MLP); %Menor é melhor
% DesvioPadrao_MLP = std(MSE_MLP); %Menor é melhor
% 
% RMSE_Medio_RB = mean(MSE_RB); %Menor é melhor
% DesvioPadrao_RB = std(MSE_RB); %Menor é melhor
% 
% RMSE_Medio_SVM = mean(MSE_SVM); %Menor é melhor
% DesvioPadrao_SVM = std(MSE_SVM); %Menor é melhor
% 
% RMSE_Medio_LM = mean(MSE_LM); %Menor é melhor
% DesvioPadrao_LM = std(MSE_LM); %Menor é melhor


% plot(performance_SVM,performance_SVM);

%RespostaFinal = RMSE_Medio +o- o DesvioPadrao e R
% nomes = categorical({'MLP','RB','MSV','LM'})
% RMSE_Medio{j} =[RMSE_Medio_MLP; RMSE_Medio_RB; RMSE_Medio_SVM; RMSE_Medio_LM];
% R_Medio{j} =[RMSE_Medio_MLP; RMSE_Medio_RB; RMSE_Medio_SVM; RMSE_Medio_LM];
% STD_Medio{j} =[DesvioPadrao_MLP; DesvioPadrao_RB; DesvioPadrao_SVM; DesvioPadrao_LM];

%   figure()
%   grid on, hold on
%   bar(nomes,RMSE_Medio);
%   title(['Erro médio quadrático - usando matriz cruzada'])
 
 
end

 
TodosDados = cat(1,MSE_GRNN, R_GRNN, STD_GRNN, MSE_LM, R_LM, STD_LM, MSE_MLP, R_MLP, STD_MLP, MSE_RB, R_RB, STD_RB, MSE_SVM, R_SVM, STD_SVM);