clc;
clear all;
close all;
data=csvread('G:\Project\support_vector_machine\date2.csv');
disp(length(data))

% Z-score Normalization*

data(:,1:end-1)=zscore(data(:,1:end-1));


[train,test] = holdout(data,60);

% Test set
Xtest=test(:,1:end-1);
Ytest=test(:,end);

% Training set
X=train(:,1:end-1);
Y=train(:,end);
 
% plotting data 

figure
hold on
scatter(X(Y==1,1),X(Y==1,2),'+b')
scatter(X(Y==-1,1),X(Y==-1,2),'.r')
xlabel('{x_1}')
ylabel('{x_2}')
legend('+ve','-ve')
title('classification')
hold off
% finding  with linear Kernel
%
fm_=[];
for c=[0.1,1,2,5,7,10]
     
     % alpha
     alpha = Lagrange(X,Y,c);
     
     % Possible support vectors
     Xs=X(alpha>0,:); Ys=Y(alpha>0);
     
     % weights
     W=(alpha(alpha>0).*Ys)'*Xs;
     
     % bias
     bias=mean(Ys-(Xs*W'));
     
     
     f=sign(Xtest*W'+bias);
     
     % confusion matrix
     fm= confusion_mat(Ytest,f);
     fm_=[fm_; c fm];    
end

% After Cross-Validation, Optimal 'c' value- Yeilding Best Performance (F-measure)
%
[max_fm, indx]=max(fm_(:,2));
c_optimal=fm_(indx,1)
% Final Model
% * _*Alpha*_
%
alpha = Lagrange(X,Y,c_optimal);
Xs=X(alpha>0,:); Ys=Y(alpha>0);
Support_vectors=size(Xs,1);
% 
% * _*Weights*_

W=(alpha(alpha>0).*Ys)'*Xs
% 
% * _*Bias*_

bias=mean(Ys-(Xs*W'))    
% 
% * _*f~ (Predicted labels)*_


f=sign(Xtest*W'+bias);
% 
% * _*Performace Measure*_

[F_measure, Accuracy] = confusion_mat(Ytest,f)


ft=X*W'+bias;

Support_vectors;
% Plotting the Hyperplane
%
figure
hold on

scatter(X(Y==1,1),X(Y==1,2),'b')
scatter(X(Y==-1,1),X(Y==-1,2),'r')
scatter(Xs(Ys==1,1),Xs(Ys==1,2),'.b')
scatter(Xs(Ys==-1,1),Xs(Ys==-1,2),'.r')

%plotting Hyperplane and Margins
syms x
fn=vpa((-bias-W(1)*x)/W(2),4);
fplot(fn,'Linewidth',2);
fn1=vpa((1-bias-W(1)*x)/W(2),4);
fplot(fn1,'--');
fn2=vpa((-1-bias-W(1)*x)/W(2),4);
fplot(fn2,'--');

%axis([-1 1.2 -1 1])
xlabel('X_1')
ylabel('X_2')
title(' CLASSIFICATION USING SVM')
legend('+ve','-ve','support vector (positive)','support vector (negative)','Decision Boundry','Location','southeast')
hold off
Accuracy=Accuracy*100
