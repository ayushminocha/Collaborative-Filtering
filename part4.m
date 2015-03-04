load ~/Documents/239AS/ml-100k/u.data;
Rmat = zeros(943,1682);
for i=1:100000
    Rmat(u(i,1),u(i,2)) = u(i,3);
end

Wmat = zeros(943,1682);
Wmat(find(Rmat > 0)) = 1;

option = struct();
option.dis = false;


index = randperm(100000);
steps = [1,10001,20001,30001,40001,50001,60001,70001,80001,90001];

pr_part1 = zeros(10,24,3);
rec_part1 = zeros(10,24,3);
pr_part2 = zeros(10,24,3);
rec_part2 = zeros(10,24,3);

Rmat_thresholded = Rmat;
Rmat_thresholded(find(Rmat <= 3)) = -1;
Rmat_thresholded(find(Rmat > 3)) = 1;

Rmat_thresholded_2 = Wmat;
Rmat_thresholded_2(find(Wmat == 0)) = -1;

k = [10,50,100];

for itr=1:length(k)
    
    for cross_validate = 1:10
        
        Rmat_part2 = Rmat;
        Rmat_part1 = Rmat;
        
        for st = steps(cross_validate):steps(cross_validate)+10000-1
            ind = index(st);
            Rmat_part2(u(ind,1),u(ind,2)) = nan;
            Rmat_part1(u(ind,1),u(ind,2)) = nan;
        end        
        
        [U_1,V_1] = wnmfrule(Rmat_part1,k(itr),option);
        UV_1 = U_1*V_1;
        
        th = 1;
        for thresh = 0.2:0.2:4.8
            UV_1_thresholded = UV_1;
            UV_1_thresholded(find(UV_1 <= thresh)) = -1;
            UV_1_thresholded(find(UV_1 > thresh)) = 1;
            
            gt = [];
            dt = [];
            for st = steps(cross_validate):steps(cross_validate)+10000-1
                ind = index(st);
                i = u(ind,1);
                j = u(ind,2);
                gt = [gt Rmat_thresholded(i,j)];
                dt = [dt UV_1_thresholded(i,j)];
            end
            
            temp1 = dt-gt;
            temp2 = dt+gt;
            tp = length(find(temp2 == 2));
            tn = length(find(temp2 == -2));
            fp = length(find(temp1 == 2));
            fn = length(find(temp1 == -2));
            
            pr_part1(cross_validate,th,itr) = tp/(tp+fp);
            rec_part1(cross_validate,th,itr) = tp/(tp+fn);
            th = th+1;
                      
        end
        
    end
end

mean_pr = mean(pr_part1,1);
mean_rec = mean(rec_part1,1);
mean_pr_2 = mean(pr_part2,1);
mean_rec_2 = mean(rec_part2,1);
save('part4_full.mat');