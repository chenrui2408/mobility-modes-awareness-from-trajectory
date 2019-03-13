function [ SetOfClusters, RD, CD, order ] = cluster_optics(points, minpts, epsilon)


disp('Calculating reachability for all points.');
tic;
[RD,CD,order]=optics(points,minpts);
toc;

disp('Computing clusters.');
tic;
mib = 0;
i = 1;
SetOfSteepDownAreas = struct();
SetOfClusters = struct();

while i < size(points,1)-1
    mib = max([mib, RD(order(i))]);   
    if RD(order(i))*(1-epsilon) >= RD(order(i+1))
        % update mib values and filter down areas
        for k=2:size(SetOfSteepDownAreas,2)
            SetOfSteepDownAreas(k).mib = max(RD(order((SetOfSteepDownAreas(k).end+1):i)));
        end
        k=2;
        while k<=size(SetOfSteepDownAreas,2)
            if RD(order(SetOfSteepDownAreas(k).start))*(1-epsilon) < mib
                if k==size(SetOfSteepDownAreas,2)
                    SetOfSteepDownAreas = SetOfSteepDownAreas(1:k-1);
                else
                    SetOfSteepDownAreas = SetOfSteepDownAreas([1:k-1, k+1:size(SetOfSteepDownAreas,2)]);
                end
            else
                k = k+1;
            end
        end
          
        newD = size(SetOfSteepDownAreas,2)+1;
        SetOfSteepDownAreas(newD).start = i;
        SetOfSteepDownAreas(newD).mib = 0;
        
        % find end of downward area
        while i < size(points,1)-1
            if RD(order(i))*(1-epsilon) >= RD(order(i+1))
                i = i+1;
            else
                j = i;
                while j < size(points,1)-1
                    if or(j-i>minpts, RD(order(j)) < RD(order(j+1)))
                        % if the downward area that isn't steep is longer than minpts, or no longer downward
                        j=-1;
                        break;
                    elseif RD(order(j))*(1-epsilon) >= RD(order(j+1))
                        % if it is a steepdownward area
                        break;
                    else
                        j = j+1;
                    end
                end
                
                if or(j == -1, j == size(points,1)-1)
                    % end of downward area
                    break;
                else
                    i = j;
                end
            end
        end
        
        SetOfSteepDownAreas(newD).end = i-1;
        mib = RD(order(i));
        
    elseif RD(order(i)) <= RD(order(i+1))*(1-epsilon)
        % Up area
        upAreaStart = i;
        % update mib values and filter down areas
        for k=2:size(SetOfSteepDownAreas,2)
            SetOfSteepDownAreas(k).mib = max(RD(order(SetOfSteepDownAreas(k).end:i)));
        end
        k=2;
        while k<=size(SetOfSteepDownAreas,2)
            if RD(order(SetOfSteepDownAreas(k).start))*(1-epsilon) < mib
                if k==size(SetOfSteepDownAreas,2)
                    SetOfSteepDownAreas = SetOfSteepDownAreas(1:k-1);
                else
                    SetOfSteepDownAreas = SetOfSteepDownAreas([1:k-1, k+1:size(SetOfSteepDownAreas,2)]);
                end
            else
                k = k+1;
            end
        end
        
        
        % find end of upward area
        while i < size(points,1)-1
            if RD(order(i)) <= RD(order(i+1))*(1-epsilon)
                i = i+1;
            else
                j = i;
                while j < size(points,1)-1
                    if or(j-i>minpts, RD(order(j)) > RD(order(j+1)))
                        % if the upward area that isn't steep is longer than minpts, or no longer upward
                        j=-1;
                        break;
                    elseif RD(order(j)) <= RD(order(j+1))*(1-epsilon)
                        % if it is a steepdownward area
                        break;
                    else
                        j = j+1;
                    end
                end
                
                if or(j == -1, j== size(points,1)-1)
                    % end of downward area
                    break;
                else
                    i = j;
                end
            end
        end
        
        mib = RD(order(i));
        
        for k=2:size(SetOfSteepDownAreas,2)
            if RD(order(i))*(1-epsilon) > SetOfSteepDownAreas(k).mib
                if and(RD(order(SetOfSteepDownAreas(k).start)) >= RD(upAreaStart) , RD(order(SetOfSteepDownAreas(k).end)) <= RD(order(i)))
                    if abs(RD(order(SetOfSteepDownAreas(k).start))-RD(order(i))) <= epsilon*max(RD(order(SetOfSteepDownAreas(k).start)),RD(order(i)))
                        % condition a
                        clusterStart = SetOfSteepDownAreas(k).start;
                        clusterEnd = i;
                    elseif RD(order(SetOfSteepDownAreas(k).start))*(1-epsilon) > RD(order(i))
                        % condition b
                        tmp = abs(RD(SetOfSteepDownAreas(k).start:SetOfSteepDownAreas(k).end)-RD(order(i)));
                        [~, clusterStart] = min(tmp); %index of closest value
                        clusterStart = clusterStart+SetOfSteepDownAreas(k).start-1;
                        clusterEnd = i;
                    elseif RD(order(SetOfSteepDownAreas(k).start)) < RD(order(i))*(1-epsilon)
                        % condition c
                        clusterStart = SetOfSteepDownAreas(k).start;
                        tmp = abs(RD(upAreaStart:i)-RD(order(SetOfSteepDownAreas(k).start)));
                        [~, clusterEnd] = min(tmp); %index of closest value
                        clusterEnd = clusterEnd+upAreaStart;
                    else
                        error('ERROR\n');
                    end
                    
                    if abs(clusterEnd - clusterStart) >= minpts
                        newD = size(SetOfClusters,2)+1;
                        SetOfClusters(newD).start = clusterStart;
                        SetOfClusters(newD).end = clusterEnd;
                    end
                end
            end
        end
        
    else
        i = i+1;
    end
end

SetOfClusters = SetOfClusters(2:size(SetOfClusters,2));
toc;

end
