load ~/Documents/239AS/ml-100k/u.data;

Rmat = zeros(943,1682);
for i=1:100000
    Rmat(u(i,1),u(i,2)) = u(i,3);
end

Wmat = zeros(943,1682);
Wmat(find(Rmat > 0)) = 1;

tempRmat = Rmat;
tempRmat(find(Rmat == 0)) = nan;

%Rmat_dup = Rmat;
%Rmat_dup(isnan(Rmat)) = 0;

option = struct();
option.dis = false;

k = [10,50,100];
LSE = zeros(length(k),1);
LSE2 = zeros(length(k),1);
finalResidual = zeros(length(k),1);
for itr=1:length(k)
    
    [U,V,numIter,tElapsed,finalResidual(itr)] = wnmfrule_modified_part2(tempRmat,k(itr),option);
    UV = U*V;
        
    LSE(itr) = sqrt(sum(sum((Rmat .* (Wmat - UV)).^2)))
    LSE2(itr) = sqrt(sum(sum(Rmat .* (Wmat - UV).^2)))    
end