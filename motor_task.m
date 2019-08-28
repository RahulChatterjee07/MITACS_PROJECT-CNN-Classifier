% 

n=1:151;
y1=n;
y1 = reshape([y1],[],1)

%%
p=[1];
n=length(p);k=1;j=1; 
for i=1:151
    while j<=n 
        y2(k)=p(j); 
        k=k+1; 
        j=j+1; 
    end
    j=1; 
end

y2 = reshape([y2],[],1)
%%
m = randi([2 4],1,75);
n=m/4;
y3 = reshape([n; zeros(size(n))],[],1)

%%
x=[0 1];
n=length(x);k=1;j=1; 
for i=1:75 
    while j<=n 
        y4(k)=x(j); 
        k=k+1; 
        j=j+1; 
    end
    j=1; 
end
y4=[y4;0]
y4= reshape([y4],[],1)

%%
x2=[0 4000];
n=length(x2);k=1;j=1; 
for i=1:75 
    while j<=n 
        y5(k)=x2(j); 
        k=k+1; 
        j=j+1; 
    end
    j=1; 
end
y5=[y5;0]
y5=reshape([y5],[],1)
%%
r = randi([8000 10000],1,75);
y6 = reshape([r; zeros(size(r))],[],1)
y6=[3000;0;y6];
y6(end)=[];
y6
%%

x1=[0 0.15];
n=length(x1);
k=1;
j=1; 
for i=1:75 
    while j<=n 
        y7(k)=x(j); 
        k=k+1; 
        j=j+1; 
    end
    j=1; 
end
y7=[y7;0]
y7 = reshape([y7],[],1)
%%
p1=[0];
n=length(p);k=1;j=1; 
for i=1:151
    while j<=n 
        y8(k)=p1(j); 
        k=k+1; 
        j=j+1; 
    end
    j=1; 
end
y8 = reshape([y8],[],1)

%%
p=[1];
n=length(p);k=1;j=1; 
for i=1:151
    while j<=n 
        y9(k)=p(j); 
        k=k+1; 
        j=j+1; 
    end
    j=1; 
end
y9 = reshape([y9],[],1)
%%
T = table(y1,y2,y3,y4,y5,y6,y7,y8,y9);
writetable(T,'myData1.txt','Delimiter','')  
type 'myData1.txt'