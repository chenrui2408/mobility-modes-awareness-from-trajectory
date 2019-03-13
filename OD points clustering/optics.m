

function [RD,CD,order]=optics(x,k)

[m,~]=size(x);
CD=zeros(1,m);
RD=ones(1,m)*10^10;

% Calculate Core Distances
for i=1:m	
    D=sort(x(i,:));%sort(dist(x(i,:),x));
    CD(i)=D(k+1);  
end

order=[];
seeds=[1:m];

ind=1;

while ~isempty(seeds)
    ob=seeds(ind);        
    seeds(ind)=[]; 
    order=[order ob];
    mm=max([ones(1,length(seeds))*CD(ob);x(ob,seeds)]);%dist(x(ob,:),x(seeds,:))
    ii=(RD(seeds))>mm;
    RD(seeds(ii))=mm(ii);
    [i1 ind]=min(RD(seeds));
end   

RD(1)=max(RD(2:m))+.1*max(RD(2:m));


function [D]=dist(i,x)

[m,n]=size(x);
D=(sum((((ones(m,1)*i)-x).^2)'));

if n==1
   D=abs((ones(m,1)*i-x))';
end
