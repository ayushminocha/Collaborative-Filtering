load ~/Documents/239AS/ml-100k/u.data;

Rmat = zeros(943,1682);
for i=1:100000
    Rmat(u(i,1),u(i,2)) = u(i,3);
end

Wmat = zeros(943,1682);
Wmat(find(Rmat > 0)) = 1;

[num_user num_movie] = size(Rmat);

lambda = [0.01,0.1,1];

option = struct();
option.dis = true;


k = [10,50,100];

err = zeros(length(k),length(lambda));

Rmat_2 = Rmat;
Rmat_2(find(Rmat == 0)) = nan;
Wmat_2 = Wmat;

err_2 = zeros(length(k),length(lambda));

for lb = 1:length(lambda)
    
    for itr=1:length(k)
        
        [U,V] = wnmfrule_modified_part5(Rmat_2,k(itr),lambda(lb),option);
        UV = U*V;
        
        err(itr,lb) = sqrt(sum(sum((Wmat .* (Rmat - UV)).^2))); 
    end
    
    %%Second part
    
    for itr=1:length(k)
        
        [U,V] = wnmfrule_modified_part5_part2(Rmat_2,k(itr),lambda(lb),option);
        UV = U*V;
        
        
        err_2(itr,lb) = sqrt(sum(sum(Rmat .* (Wmat - UV).^2)));        
    end
end

