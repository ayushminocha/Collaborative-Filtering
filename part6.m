load ~/Documents/239AS/ml-100k/u.data;
numUser = 943;
numMovies = 1682;
Rmat = zeros(numUser,numMovies);
for i=1:100000
    Rmat(u(i,1),u(i,2)) = u(i,3);
end

Wmat = zeros(943,1682);
Wmat(find(Rmat > 0)) = 1;

[num_user num_movie] = size(Rmat);
lambda = [0.01,0.1,1];
%lambda = 0.1;

option = struct();
option.dis = false;


k = [10,50,100];
%k = 10;
err = zeros(length(k),length(lambda));

Rmat_2 = Rmat;
Rmat_2(find(Rmat == 0)) = nan;

index = randperm(100000);
steps = [1,10001,20001,30001,40001,50001,60001,70001,80001,90001];

Rmat_thresholded = Rmat;
Rmat_thresholded(find(Rmat <= 3)) = 0;
Rmat_thresholded(find(Rmat > 3)) = 1;
Rmat_thresholded(isnan(Rmat_2)) = nan;
numCrossValidate = 10;
pre = zeros(numUser,numCrossValidate,length(k),length(lambda));
tempHit = -1*ones(numUser,1682);
tempFalse = -1*ones(numUser,1682);

finalHitRate = inf(numCrossValidate,length(k),length(lambda),1682);
finalFalseRate = inf(numCrossValidate,length(k),length(lambda),1682);

for lb = 1:length(lambda)

    for itr=1:length(k)
        
        for cross_validate = 1:10
        
        tempRmat = Rmat_2;
        testRmat = ones(size(Rmat_2));
        
        for st = steps(cross_validate):steps(cross_validate)+10000-1
            ind = index(st);            
            tempRmat(u(ind,1),u(ind,2)) = nan;
            testRmat(u(ind,1),u(ind,2)) = nan;
        end        
        
        
        [U,V] = wnmfrule_modified_part5_part2(tempRmat,k(itr),lambda(lb),option);
        UV = U*V;
        
        tempW = isnan(testRmat);
        predMat = -1*inf(size(UV));
        predMat(tempW) = UV(tempW);
        testLiked = nan(size(predMat));
        testLiked(tempW) = Rmat_thresholded(tempW);
        
        for user=1:numUser
            [sortedValues,sortIndex] = sort(predMat(user,:),'descend');
            maxIndex = sortIndex(1:5);
            total = 0;
            for mi=1:5
                if(~isnan(Rmat_thresholded(user,maxIndex(mi))))
                    total = total+1;
                    if(Rmat_thresholded(user,maxIndex(mi)))
                        pre(user,cross_validate,itr,lb) = pre(user,cross_validate,itr,lb)+1;
                    end                  
                end
            end
            if(total)
                pre(user,cross_validate,itr,lb) = pre(user,cross_validate,itr,lb)/total;
            else
                pre(user,cross_validate,itr,lb) = 0;
            end
            
            testLen = length(find(predMat(user,:)~=-inf));
            
            hitRate = zeros(testLen,1);
            falseRate = zeros(testLen,1);
            for topL=1:testLen
                if(~isnan(Rmat_thresholded(user,sortIndex(topL))))                    
                    if(Rmat_thresholded(user,sortIndex(topL)))
                        hitRate(topL) = hitRate(topL)+1;
                    else
                        falseRate(topL) = falseRate(topL)+1;
                    end
                end
            end
            
            
            if(testLen && length(find(testLiked(user,:) == 1)) && length(find(testLiked(user,:) == 0)))
                tempHit(user,1) = hitRate(1)/(length(find(testLiked(user,:) == 1)));
                tempFalse(user,1) = falseRate(1)/(length(find(testLiked(user,:) == 0)));
                
                for topL = 2:testLen
                    hitRate(topL) = hitRate(topL)+hitRate(topL-1);
                    falseRate(topL) = falseRate(topL)+falseRate(topL-1);
                    tempHit(user,topL) = hitRate(topL)/(length(find(testLiked(user,:) == 1)));
                    tempFalse(user,topL) = falseRate(topL)/(length(find(testLiked(user,:) == 0)));
                end
            end
            
        end 
        meanHit = zeros(1682,1);
        meanFalse = zeros(1682,1);
        
        for movies = 1:1682
            tempp = tempHit(:,movies);
            meanHit(movies) = mean(tempp(tempp~=-1));
            
            tempp = tempFalse(:,movies);
            meanFalse(movies) = mean(tempp(tempp~=-1));
        end
        
        for topL = 1:testLen
            finalHitRate(cross_validate,itr,lb,topL) = meanHit(topL);
            finalFalseRate(cross_validate,itr,lb,topL) = meanFalse(topL);
        end
        
        end      
    end
end

