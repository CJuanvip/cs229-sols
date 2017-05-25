A = double(imread('mandrill-small.tiff'));
imshow(uint8(round(A)));

% K-means initialization
k = 16;
initmu = zeros(k,3);
for l=1:k,
    i = random(’unid’, size(A, 1), 1, 1);
    j = random(’unid’, size(A, 2), 1, 1);
    initmu(l,:) = double(permute(A(i,j,:), [3 2 1])’);
end;

% Run K-means
mu = initmu;
for iter = 1:200, % usually converges long before 200 iterations
    newmu = zeros(k,3);
    nassign = zeros(k,1);
    for i=1:size(A,1),
        for j=1:size(A,2),
            dist = zeros(k,1);
            for l=1:k,
                d = mu(l,:)’-permute(A(i,j,:), [3 2 1]);
                dist(l) = d’*d;
            end;
            [value, assignment] = min(dist);
            nassign(assignment) = nassign(assignment) + 1;
            newmu(assignment,:) = newmu(assignment,:) + ...
                permute(A(i,j,:), [3 2 1])’;
        end; 
    end;
    for l=1:k,
        if (nassign(l) > 0)
            newmu(l,:) = newmu(l,:) / nassign(l);
        end;
    end;
    mu = newmu;
end;

% Assign new colors to large image
bigimage = double(imread('mandrill-large.tiff'));
imshow(uint8(round(bigimage)));
qimage = bigimage;
for i=1:size(bigimage,1), 
    for j=1:size(bigimage,2),
        dist = zeros(k,1);
        for l=1:k,
            d = mu(l,:)’-permute(bigimage(i,j,:), [3 2 1]);
            dist(l) = d’*d;
        end;
        [value, assignment] = min(dist);
        qimage(i,j,:) = ipermute(mu(assignment,:), [3 2 1]);
    end; 
end;

imshow(uint8(round(qimage)));