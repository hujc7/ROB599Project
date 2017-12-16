files = dir('trainval/*/*_image.jpg');
for i=1:numel(files)
    idx = i;
    snapshot = [files(idx).folder, '/', files(idx).name];
    disp(snapshot)
    
    fid = fopen(strrep(snapshot, '_image.jpg', '_label.txt'),'w');
    img = imread(snapshot);
    [h_img, w_img, ~] = size(img);  % Height and width of image
    xyz = read_bin(strrep(snapshot, '_image.jpg', '_cloud.bin'));
    xyz = reshape(xyz, [numel(xyz) / 3, 3])';

    proj = read_bin(strrep(snapshot, '_image.jpg', '_proj.bin'));
    proj = reshape(proj, [4, 3])';

    try
        bbox = read_bin(strrep(snapshot, '_image.jpg', '_bbox.bin'));
    catch
        disp('[*] no bbox found.')
        bbox = single([]);
    end
    bbox = reshape(bbox, [11, numel(bbox) / 11])';

    uv = proj * [xyz; ones(1, size(xyz, 2))];
    uv = uv ./ uv(3, :);
    for k = 1:size(bbox, 1)
        b = bbox(k, :);

        n = b(1:3);
        theta = norm(n, 2);
        n = n / theta;
        R = rot(n, theta);
        t = reshape(b(4:6), [3, 1]);

        sz = b(7:9);
        [vert_3D, edges] = get_bbox(-sz / 2, sz / 2);
        vert_3D = R * vert_3D + t;

        vert_2D = proj * [vert_3D; ones(1, 8)];
        vert_2D = vert_2D ./ vert_2D(3, :);
        if b(11)==0
            x = mean(vert_2D(1,:))/w_img;
            y = mean(vert_2D(2,:))/h_img;
            w = (max(vert_2D(1,:))-min(vert_2D(1,:)))/w_img;
            h = (max(vert_2D(2,:))-min(vert_2D(2,:)))/h_img;
            fprintf(fid,'%d %f %f %f %f\n',b(10),x,y,w,h);
        end
    end
    fclose(fid);
end


function [v, e] = get_bbox(p1, p2)
v = [p1(1), p1(1), p1(1), p1(1), p2(1), p2(1), p2(1), p2(1)
    p1(2), p1(2), p2(2), p2(2), p1(2), p1(2), p2(2), p2(2)
    p1(3), p2(3), p1(3), p2(3), p1(3), p2(3), p1(3), p2(3)];
e = [3, 4, 1, 1, 4, 4, 1, 2, 3, 4, 5, 5, 8, 8
    8, 7, 2, 3, 2, 3, 5, 6, 7, 8, 6, 7, 6, 7];
end


function R = rot(n, theta)
K = [0, -n(3), n(2); n(3), 0, -n(1); -n(2), n(1), 0];
R = eye(3) + sin(theta) * K + (1 - cos(theta)) * K^2;
end


function data = read_bin(file_name)
id = fopen(file_name, 'r');
data = fread(id, inf, 'single');
fclose(id);
end
