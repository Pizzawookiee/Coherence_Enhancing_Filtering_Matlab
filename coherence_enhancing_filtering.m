
%to-do:
%fix index error when using bilinear sampling on edge of image

function image = coherence_enhancing_filtering(image, sigma_s, steps, sigma_g, sigma_i, radius, sensitivity)
%default sigma_s = 6
%default steps = 2
%default sigma_g = 1.5
%default radius = 2
%default sensitivity = 0.01

p = 32;
%pad by p, then remove padding to prevent strange edge behavior
image = padarray(image, [p,p], 'symmetric', 'both');
[height, width, numChannels] = size(image);

[E,F,G] = structure_tensor(image, height, width, numChannels);
[E,F,G] = relaxation(E,F,G,10, height, width); %why is relaxation so slow?
[E,F,G] = structure_tensor_smoothing(E,F,G,5); %is this the right smoothing?

for i=1:steps
    image = line_integral_convolution(E,F,G,sigma_s,image, height, width, numChannels);
end

grey_smoothed_img = uint8(im2gray(image));
%assume need new structure tensor for grey_img
[E,F,G] = structure_tensor(grey_smoothed_img, height, width, 1); %1 channel for greyscale
[E,F,G] = structure_tensor_smoothing(E,F,G,5); %is this the right smoothing?
image = gradient_shock_filter(E,F,G,sigma_g,sigma_i,image, grey_smoothed_img, height, width, numChannels, radius, sensitivity);
image = line_integral_convolution(E,F,G,1.5, image, height, width, numChannels);
%remove padding
image = image(p:end-p, p:end-p, :);
image = cast(image, 'uint8');
end

function new_img = gradient_shock_filter(E,F,G,sigma_g, sigma_i, image, grey_img, height, width, numChannels, radius, sensitivity)
new_img = uint8(zeros([height, width, numChannels]));
if sigma_i > 0
    grey_img = imgaussfilt(grey_img, sigma_i);
end

%create gaussian conv filter
filter = zeros(1,1,1,2*radius+1);
filter(:,:,:,1) = -1/(sqrt(2*pi)*sigma_g);
for i = 1:radius
    filter(:,:,:,2*i) = -1/(sqrt(2*pi)*sigma_g) + exp(-i.^2/(2 * sigma_g.^2));
    filter(:,:,:,2*i+1) = -1/(sqrt(2*pi)*sigma_g) + exp(-i.^2/(2 * sigma_g.^2));
end

%E_interpolator = griddedInterpolant(E);
%F_interpolator = griddedInterpolant(F);
%G_interpolator = griddedInterpolant(G);
%img_interpolator = griddedInterpolant(double(image));
%grey_img_interpolator = griddedInterpolant(double(grey_img));
%grey_img_interpolator
%c=1;
for y = 1:height
    for x = 1:width
        new_img(y,x,:) = gradient_shock_filter_at_point(y,x,E,F,G,image, grey_img, height, width, numChannels, radius, filter, sensitivity);
        %new_img(y,x,:) = gradient_shock_filter_at_point(y,x,E_interpolator,F_interpolator,G_interpolator,img_interpolator, grey_img_interpolator, height, width, numChannels, radius, filter, sensitivity);
        %c
        %c = c+1;
    end
end

end

function new_pixel = gradient_shock_filter_at_point(img_y, img_x, E,F,G,image, grey_img, height, width, numChannels, radius, filter, sensitivity)
[major_e_vec,~, ~] = process_structure_tensor_at_point(img_y, img_x, E, F, G, height, width);
result = zeros(1,1,1,2 * radius + 1);
locations = zeros(2, 2*radius + 1); %store locations here to index from image at the very end
isValid = zeros(1,2*radius+1);
result(:,:,:,1) = sample_img_pixel(img_y, img_x, grey_img, height, width, 1);
%result(:,:,:,1) = grey_img(img_y, img_x);
locations(:,1) = [img_y, img_x];

forward_change = major_e_vec;
forward_y = img_y + forward_change(1);
forward_x = img_x + forward_change(2);

backward_change = -major_e_vec;

backward_y = img_y + backward_change(1);
backward_x = img_x + backward_change(2);
[forward_pixel, forward_valid] = sample_img_pixel(forward_y, forward_x, grey_img, height, width, 1);
[backward_pixel, backward_valid] = sample_img_pixel(backward_y, backward_x, grey_img, height, width, 1);
result(:,:,:,2) = forward_pixel * forward_valid;
result(:,:,:,3) = backward_pixel * backward_valid;
%result(:,:,:,2) = grey_img(forward_y, forward_x);
%result(:,:,:,3) = grey_img(backward_y, backward_x);
isValid(:,2) = forward_valid;
isValid(:,3) = backward_valid;
locations(:,2) = [forward_y, forward_x];
locations(:,3) = [backward_y, backward_x];

for i = 2:radius
    [next_forward_change,~, ~] = process_structure_tensor_at_point(forward_y, forward_x, E,F,G, height, width);
    forward_change = sign(dot(forward_change, next_forward_change)) * next_forward_change;
    forward_y = forward_y + forward_change(1);
    forward_x = forward_x + forward_change(2);
    [next_backward_change,~, ~] = process_structure_tensor_at_point(backward_y, backward_x, E,F,G, height, width);
    backward_change = sign(dot(backward_change, next_backward_change)) * next_backward_change;

    backward_y = backward_y + backward_change(1);

    backward_x = backward_x + backward_change(2);
    [forward_pixel, forward_valid] = sample_img_pixel(forward_y, forward_x, grey_img, height, width, 1);
    [backward_pixel, backward_valid] = sample_img_pixel(backward_y, backward_x, grey_img, height, width, 1);



    result(:,:,:,2*i) = forward_pixel * forward_valid;
    result(:,:,:,2*i+1) = backward_pixel * backward_valid;

    %result(:,:,:,2*i) = grey_img(forward_y, forward_x);
    %result(:,:,:,2*i+1) = grey_img(backward_y, backward_x);
    locations(:,2*i) = [forward_y, forward_x];
    locations(:,2*i+1) = [backward_y, backward_x];
    isValid(:,2*i) = forward_valid;
    isValid(:,2*i+1) = backward_valid;
end
summed_result = sum(result .* filter, 4);
if summed_result > sensitivity
local_min_index = 1;
local_min = result(:,:,:,1);
for k = 2:2*radius+1
   temp = result(:,:,:,k);
   if temp < local_min && isValid(:,k)
       local_min_index = local_min_index + 1;
   end

end
indices = locations(:,local_min_index);
%{
new_pixel = zeros(1,1,numChannels);
for h = 1:numChannels
    new_pixel(:,:,h) = uint8(image(indices(1), indices(2), h));
end
%}
new_pixel = sample_img_pixel(indices(1), indices(2), image, height, width, numChannels);
else
    if summed_result < -sensitivity
        local_max_index = 1;
        local_max = result(:,:,:,1);
        for k = 2:2*radius+1
           temp = result(:,:,:,k);
           if temp > local_max && isValid(:,k)
               local_max_index = local_max_index + 1;
           end
        end
        indices = locations(:,local_max_index);
        %{
        new_pixel = zeros(1,1,numChannels);
        for h = 1:numChannels
            new_pixel(:,:,h) = uint8(image(indices(1), indices(2), h));
        end
        %}
        new_pixel = sample_img_pixel(indices(1), indices(2), image, height, width, numChannels);

    else
        new_pixel = sample_img_pixel(img_y, img_x, image, height, width, numChannels);
        %{
        new_pixel = zeros(1,1,numChannels);
        for h = 1:numChannels
            new_pixel(:,:,h) = uint8(image(img_y, img_x, h));
        end
        %}
    end
end





end







function new_img = line_integral_convolution(E,F,G,sigma_s, image, height, width, numChannels)
%speed up by calculating filters once using sigma_s
%how to store arrays of multiple sizes?
new_img = uint8(zeros([height, width, numChannels]));
%E_interpolator = griddedInterpolant(E);
%F_interpolator = griddedInterpolant(F);
%G_interpolator = griddedInterpolant(G);
%img_interpolator = griddedInterpolant(double(image));

%c = 1;
for y = 1:height
    for x = 1:width
        new_img(y,x,:) = line_integral_convolution_at_point(y,x,E,F,G,sigma_s, image, height, width, numChannels);
        %new_img(y,x,:) = line_integral_convolution_at_point(y,x,E_interpolator,F_interpolator,G_interpolator,sigma_s, img_interpolator, height, width, numChannels);
        %c
        %c = c+1;
    end
end

end

function new_pixel = line_integral_convolution_at_point(img_y, img_x, E, F, G, sigma_s, image, height, width, numChannels)
%uses sigma_s parameter from paper, this controls smoothing
[~,minor_e_vec, A] = process_structure_tensor_at_point(img_y, img_x, E, F, G, height, width);
adapted_sigma = 0.25*sigma_s * (1+A).^2;
l = ceil(2 * adapted_sigma);
result = zeros(1,1,numChannels, 1+2*l);
initial_weight = 1/(adapted_sigma *sqrt(2 * pi)) * exp(-0.5 *(0-0).^2/(adapted_sigma.^2));
normalization_factor = initial_weight;

%first_step;
result(:,:,:,1) = sample_img_pixel(img_y, img_x, image, height, width, numChannels) * initial_weight;
%{
init_pixel = zeros(1,1,numChannels);
for h = 1:numChannels
    init_pixel(:,:,h) = image(img_y, img_x, h);
end

result(:,:,:,1) = init_pixel * initial_weight;
%}


%use runge-kutta second order
forward_change = minor_e_vec;
%forward_y = img_y + forward_change(1) * (img_y + 0.5 * (forward_change(1) * img_y));
%forward_x = img_x + forward_change(2) * (img_x + 0.5 * (forward_change(2) * img_x));
forward_y = img_y + forward_change(1) + 0.5 * forward_change(1).^2;
forward_x = img_x + forward_change(2) + 0.5 * forward_change(2).^2;
backward_change = -minor_e_vec;
%backward_y = img_y + backward_change(1) * (img_y + 0.5 * (backward_change(1) * img_y));
%backward_x = img_x + backward_change(2) * (img_x + 0.5 * (backward_change(2) * img_x));
backward_y = img_y + backward_change(1) + 0.5 * backward_change(1).^2;
backward_x = img_x + backward_change(2) + 0.5 * backward_change(2).^2;

%{
forward_pixel = zeros(1,1,numChannels);
for h = 1:numChannels
    forward_pixel(:,:,h) = image(forward_y, forward_x, h);
end
backward_pixel = zeros(1,1,numChannels);
for h = 1:numChannels
    backward_pixel(:,:,h) = image(backward_y, backward_x, h);
end
%}

[forward_pixel, forward_valid] = sample_img_pixel(forward_y, forward_x, image, height, width, numChannels);
[backward_pixel, backward_valid] = sample_img_pixel(backward_y, backward_x, image, height, width, numChannels);
%forward_weight = normpdf(1, 0, adapted_sigma);
%backward_weight = normpdf(-1,0,adapted_sigma);

forward_weight = 1/(adapted_sigma *sqrt(2 * pi)) * exp(-0.5 *(1-0).^2/(adapted_sigma.^2));
backward_weight = 1/(adapted_sigma *sqrt(2 * pi)) * exp(-0.5 *(-1-0).^2/(adapted_sigma.^2));
result(:,:,:,2) = forward_weight * forward_pixel * forward_valid;
result(:,:,:,3) = backward_weight * backward_pixel * backward_valid;
%result(:,:,:,2) = forward_weight * forward_pixel;
%result(:,:,:,3) = backward_weight * backward_pixel;
normalization_factor = normalization_factor + forward_weight * forward_valid + backward_weight * backward_valid;
%normalization_factor = normalization_factor + forward_weight + backward_weight;


for i = 2:l
    
    [~,next_forward_change, ~] = process_structure_tensor_at_point(forward_y, forward_x, E,F,G, height, width);
    forward_change = sign(dot(forward_change, next_forward_change)) * next_forward_change;
    %forward_y = forward_y + forward_change(1) * (forward_y + 0.5 * (forward_change(1) * forward_y));
    forward_y = forward_y + forward_change(1) + 0.5 * forward_change(1).^2;
    %forward_x = forward_x + forward_change(2) * (forward_x + 0.5 * (forward_change(2) * forward_x));
    forward_x = forward_x + forward_change(2) + 0.5 * forward_change(2).^2;
    [~,next_backward_change, ~] = process_structure_tensor_at_point(backward_y, backward_x, E,F,G, height, width);
    backward_change = sign(dot(backward_change, next_backward_change)) * next_backward_change;
    %backward_y = backward_y + backward_change(1) * (backward_y + 0.5 * (backward_change(1) * backward_y));
    backward_y = backward_y + backward_change(1) + 0.5 * backward_change(1).^2;
    %backward_x = backward_x + backward_change(2) * (backward_x + 0.5 * (backward_change(2) * backward_x));]
    backward_x = backward_x + backward_change(2) + 0.5 * backward_change(2).^2;
    %{
    forward_pixel = zeros(1,1,numChannels);
    for h = 1:numChannels
        forward_pixel(:,:,h) = image(forward_y, forward_x, h);
    end
    backward_pixel = zeros(1,1,numChannels);
    for h = 1:numChannels
        backward_pixel(:,:,h) = image(backward_y, backward_x, h);
    end
    %}
    [forward_pixel, forward_valid] = sample_img_pixel(forward_y, forward_x, image, height, width, numChannels);
    [backward_pixel, backward_valid] = sample_img_pixel(backward_y, backward_x, image, height, width, numChannels);
    forward_weight = normpdf(i, 0, adapted_sigma);
    backward_weight = normpdf(-i,0,adapted_sigma);
    forward_weight = 1/(adapted_sigma *sqrt(2 * pi)) * exp(-0.5 *(i-0).^2/(adapted_sigma.^2));
    backward_weight = 1/(adapted_sigma *sqrt(2 * pi)) * exp(-0.5 *(-i-0).^2/(adapted_sigma.^2));

    result(:,:,:,2*i) = forward_weight * forward_pixel * forward_valid;
    result(:,:,:,2*i+1) = backward_weight * backward_pixel * backward_valid;
    %result(:,:,:,2*i) = forward_weight * forward_pixel;
    %result(:,:,:,2*i+1) = backward_weight * backward_pixel;
    normalization_factor = normalization_factor + forward_weight * forward_valid + backward_weight * backward_valid;
    %normalization_factor = normalization_factor + forward_weight + backward_weight;
end
new_pixel = uint8(sum(result, 4)/normalization_factor);

end

function n = normpdf(x, mu, sigma)
%using this in lieu of statistics toolbox due to slow internet connection
n = 1/(sigma *sqrt(2 * pi)) * exp(-0.5 *(x-mu).^2/(sigma.^2));
end


function [s, b] = sample_img_pixel(img_y, img_x, image, height, width, numChannels)
%deprecated, replace with gridded Interpolant calls

if (img_y >= 1) && (img_y <= height) && (img_x >= 1) && (img_x <= width)
    b = 1;
    %nearest-neighbor version
    %{
    x1 = floor(img_x);
    x2 = ceil(img_x);
    y1 = floor(img_y);
    y2 = ceil(img_y);

    dx1 = (img_x - x1).^2;
    dx2 = (img_x - x2).^2;

    dy1 = (img_y - y1).^2;
    dy2 = (img_y - y2).^2;

    if dx1 <= dx2
        final_x = x1;
    else
        final_x = x2;
    end

    if dy1 <= dy2
        final_y = y1;
    else
        final_y = y2;
    end
    s = image(final_y, final_x, :);
    %}
    %bilinear version
    
    if (img_x == floor(img_x)) && (img_y == floor(img_y))
        s = image(img_y, img_x, :);
    else
        %s = images.internal.interp2d(image, img_y, img_x, "bilinear", 0, false);
        
        x1 = floor(img_x);
        x2 = x1 + 1;
        y1 = floor(img_y);
        y2 = y1 + 1;

        %hack to prevent indexing errors
        if x1 == width
            x2 = x1;
        end
        if y1 == height
            y2 = y1;
        end

        q11 = double(image(y1, x1, :));
        q12 = double(image(y1, x2, :));
        q21 = double(image(y2, x1, :));
        q22 = double(image(y2, x2, :));

        dx1 = (img_x - x1);
        dx2 = (x2 - img_x);
        dy1 = (img_y - y1);
        dy2 = (y2 - img_y);

        w11 = dy2 * dx2;
        w12 = dy2 * dx1;
        w21 = dy1 * dx2;
        w22 = dy1 * dx1;

        s = uint8(w11 * q11 + w12 * q12 + w21 * q21 + w22 * q22);
        

        %for i=1:numChannels
        %    s(:,:,i) = (1.0/((x2-x1)*(y2-y1)))* [x2-img_x, img_x - x1] * [q11(:,:,i), q21(:,:,i); q12(:,:,i), q22(:,:,i)] * [y2-img_y; img_y - y1];
        %end
        
        %pagemtimes is slow
        %s = pagemtimes(pagemtimes((1.0/((x2-x1)*(y2-y1)))* [x2-img_x, img_x - x1],[q11, q21; q12, q22]), [y2-img_y; img_y - y1]);
        
    end
    
else
   
   s = zeros(1,1,numChannels, 'uint8');
   b = 0;
end
end



function v = valid_pos(img_y, img_x, img_height, img_width)
%don't call this function, it's slow
if (img_y >= 1) && (img_y <= img_height) && (img_x >= 1) && (img_x <= img_width)
    v = true;
else
    v = false;
end
end

function [major_e_vec, minor_e_vec, A] = process_structure_tensor_at_point(img_y,img_x,E,F,G, img_height, img_width)
%input the genuine y,x coordinates from the image
%assumes that E,F,G still maintain the 1-pixel padding

%avoid subfunction calls!
%[e,f,g] = generate_structure_tensor_from_point(E,F,G, img_y, img_x, img_height, img_width);

%per-element bilinear sampling

if (img_y >= 1) && (img_y <= img_height) && (img_x >= 1) && (img_x <= img_width)
    
    %avoid subfunction calls for now
    if(img_x == floor(img_x)) && (img_y == floor(img_y))
        e = E(img_y, img_x);
        f = F(img_y, img_x);
        g = G(img_y, img_x);
        %[e,f,g] = generate_structure_tensor_general(E,F,G, img_y, img_x);
    else
        
        %{
        e = interp2(E, img_y, img_x);
        f = interp2(F, img_y, img_x);
        g = interp2(G, img_y, img_x);
        %}
        
        x1 = floor(img_x);
        x2 = ceil(img_x);
        y1 = floor(img_y);
        y2 = ceil(img_y);
        
        dx1 = img_x - x1;
        dx2 = x2 - img_x;
        dy1 = img_y - y1;
        dy2 = y2 - img_y;

        q11 = double(E(y1, x1));
        q12 = double(E(y1, x2));
        q21 = double(E(y2, x1));
        q22 = double(E(y2, x2));

        %e = (1.0/((x2-x1)*(y2-y1))) * [x2-img_x, img_x - x1] * [q11, q21; q12, q22] * [y2-img_y; img_y - y1];
        e = [dx2, dx1] * [q11, q21; q12, q22] * [dy2; dy1];
        q11 = double(F(y1, x1));
        q12 = double(F(y1, x2));
        q21 = double(F(y2, x1));
        q22 = double(F(y2, x2));
        %f = (1.0/((x2-x1)*(y2-y1))) * [x2-img_x, img_x - x1] * [q11, q21; q12, q22] * [y2-img_y; img_y - y1];
        f = [dx2, dx1] * [q11, q21; q12, q22] * [dy2; dy1];
        q11 = double(G(y1, x1));
        q12 = double(G(y1, x2));
        q21 = double(G(y2, x1));
        q22 = double(G(y2, x2));
        %g = (1.0/((x2-x1)*(y2-y1))) * [x2-img_x, img_x - x1] * [q11, q21; q12, q22] * [y2-img_y; img_y - y1];
        g = [dx2, dx1] * [q11, q21; q12, q22] * [dy2; dy1];
        %[e,f,g] = generate_structure_tensor_bilinear(E,F,G,img_y, img_x);
        
    end
else
    e = 1;
    f = 1;
    g = 1;
    %img_y
    %img_x
end

%using gridded interpolator
%e = E(img_y, img_x);
%f = F(img_y, img_x);
%g = G(img_y, img_x);

major_e_val = 0.5 * (e+g+sqrt((e-g).^2 + 4 * f.^2));
minor_e_val = 0.5 * (e+g-sqrt((e-g).^2 + 4 * f.^2));

minor_e_vec = [major_e_val - e; -f];
major_e_vec = [f; major_e_val - e];
%normalization
%minor_e_vec = (minor_e_vec - min(minor_e_vec(:)))/(max(minor_e_vec(:)) - min(minor_e_vec(:)));
%need unit length of 1, make sure this works?
length = norm(minor_e_vec);
if length ~= 0
    minor_e_vec = minor_e_vec / length;
    major_e_vec = major_e_vec / length;
else
    minor_e_vec = minor_e_vec / (length + 1e-12);
    major_e_vec = major_e_vec / (length + 1e-12);
end


%calculate anisotropy
e_val_diff = major_e_val - minor_e_val;
e_val_sum = major_e_val + minor_e_val;
if e_val_sum ~= 0
    A = e_val_diff / e_val_sum;
else
    A = e_val_diff / (e_val_sum + 1e-12);
end
end


function [e,f,g] = generate_structure_tensor_from_point(E,F,G, img_y, img_x, img_height, img_width)

if (img_y >= 1) && (img_y <= img_height) && (img_x >= 1) && (img_x <= img_width)
    %avoid subfunction calls for now
    if(img_x == floor(img_x)) && (img_y == floor(img_y))
        e = E(img_y, img_x);
        f = F(img_y, img_x);
        g = G(img_y, img_x);
        %[e,f,g] = generate_structure_tensor_general(E,F,G, img_y, img_x);
    else
        
        %bilinear sampling
        x1 = floor(img_x);
        x2 = ceil(img_x);
        y1 = floor(img_y);
        y2 = ceil(img_y);
        
        q11 = double(E(y1, x1));
        q12 = double(E(y1, x2));
        q21 = double(E(y2, x1));
        q22 = double(E(y2, x2));
        e = (1.0/((x2-x1)*(y2-y1))) * [x2-img_x, img_x - x1] * [q11, q21; q12, q22] * [y2-img_y; img_y - y1];
        
        q11 = double(F(y1, x1));
        q12 = double(F(y1, x2));
        q21 = double(F(y2, x1));
        q22 = double(F(y2, x2));
        f = (1.0/((x2-x1)*(y2-y1))) * [x2-img_x, img_x - x1] * [q11, q21; q12, q22] * [y2-img_y; img_y - y1];
        
        q11 = double(G(y1, x1));
        q12 = double(G(y1, x2));
        q21 = double(G(y2, x1));
        q22 = double(G(y2, x2));
        g = (1.0/((x2-x1)*(y2-y1))) * [x2-img_x, img_x - x1] * [q11, q21; q12, q22] * [y2-img_y; img_y - y1];

        %[e,f,g] = generate_structure_tensor_bilinear(E,F,G,img_y, img_x);
       
    end
else
    e = 0;
    f = 0;
    g = 0;
end
end

function [e,f,g] = generate_structure_tensor_general(E,F,G,img_y, img_x)
%deprecated
%for general case
e = E(img_y, img_x);
f = F(img_y, img_x);
g = G(img_y, img_x);

end

function [e,f,g] = generate_structure_tensor_bilinear(E,F,G,img_y,img_x)
%deprecated
%uses bilinear interpolation to guesstimate structure tensor


x1 = floor(img_x);
x2 = ceil(img_x);
y1 = floor(img_y);
y2 = ceil(img_y);

q11 = double(E(y1, x1));
q12 = double(E(y1, x2));
q21 = double(E(y2, x1));
q22 = double(E(y2, x2));
e = (1.0/((x2-x1)*(y2-y1))) * [x2-img_x, img_x - x1] * [q11, q21; q12, q22] * [y2-img_y; img_y - y1];

q11 = double(F(y1, x1));
q12 = double(F(y1, x2));
q21 = double(F(y2, x1));
q22 = double(F(y2, x2));
f = (1.0/((x2-x1)*(y2-y1))) * [x2-img_x, img_x - x1] * [q11, q21; q12, q22] * [y2-img_y; img_y - y1];

q11 = double(G(y1, x1));
q12 = double(G(y1, x2));
q21 = double(G(y2, x1));
q22 = double(G(y2, x2));
g = (1.0/((x2-x1)*(y2-y1))) * [x2-img_x, img_x - x1] * [q11, q21; q12, q22] * [y2-img_y; img_y - y1];


end

function [E,F,G] = structure_tensor_smoothing(E,F,G, sigma)
%use a gaussian filter to blur the components of the structure tensor
current_arr = cat(3,E,F,G);
l = size(current_arr,3);
for i = 1:l
    current_arr(:,:,i) = imgaussfilt(current_arr(:,:,i), sigma);
end
%for redundancy
E = current_arr(:,:,1);
F = current_arr(:,:,2);
G = current_arr(:,:,3);
end




function [E, F, G] = relaxation(E, F, G, steps, height, width)
%compute threshold conditions ahead of time

threshold = 0.002;
relax_filter = 0.25 * [0,1,0;1,0,1;0,1,0];
satisfaction = zeros(size(E));
for q = 1:steps
    
    for y = 1:height
        for x = 1:width
            if sqrt(E(y,x).^2 + G(y,x).^2 + 2 * F(y,x).^2) <= threshold
                satisfaction(y,x) = 1;
            else
                satisfaction(y,x) = 0;
            end
        end
    end
    E_temp = conv2(E, relax_filter, 'same');
    for y = 1:height
        for x = 1:width
            if satisfaction(y,x) == 1
                E(y,x) = E_temp(y,x);
            end
                
        end
    end
    F_temp = conv2(F, relax_filter, 'same');
    for y = 1:height
        for x = 1:width
            if satisfaction(y,x) == 1
                F(y,x) = F_temp(y,x);
            end
                
        end
    end
    G_temp = conv2(G, relax_filter, 'same');
    for y = 1:height
        for x = 1:width
            if satisfaction(y,x) == 1
                G(y,x) = G_temp(y,x);
            end
                
        end
    end
end


end







function [E, F, G] = structure_tensor(image, height, width, numChannels)

%z = zeros([height, width, numChannels]);

%rotational symmetric derivative
p = 0.183;
d_x = 0.5 * [p, 0, -p; 1-2*p, 0, 2*p-1; p, 0, -p];
%sobel filter for testing
%d_x = [1,0,-1; 2,0,-2; 1,0,-1];
d_y = transpose(d_x);
E_tensors = zeros(height,width,numChannels, class(p));
F_tensors = zeros(height,width,numChannels, class(p));
G_tensors = zeros(height,width,numChannels, class(p));
for n = 1:numChannels
    %process each channel separately
    c = image(:,:,n);
    f_x = conv2(c, d_x, 'same');
    f_y = conv2(c, d_y, 'same');
    %f_x = (f_x - min(f_x(:)))/(max(f_x(:))-min(f_x(:)));
    %f_y = (f_y - min(f_y(:)))/(max(f_y(:))-min(f_y(:)));
    %size(dot(f_x,f_x))
    E = f_x.^2;
    F = f_x .* f_y;
    G = f_y.^2;
    E_tensors(:,:,n) = E;
    F_tensors(:,:,n) = F;
    G_tensors(:,:,n) = G;
    %t = sqrt(f_x.^2 + f_y.^2);
    %z(:,:,n) = t;
end
E = sum(E_tensors, 3);
F = sum(F_tensors, 3);
G = sum(G_tensors, 3);
%S_tensor = [E,F;F,G];
%major eigenvalues
%eigen_1 = 0.5 * (E + G + sqrt((E-G).^2 + 4 * F.^2));
%minor eigenvalues
%eigen_2 = 0.5 * (E + G - sqrt((E-G).^2 + 4 * F.^2));
%major eigenvector (use a positional function!)
%e_major = [F;eigen_1-E];
%minor eigenvector (jkyprian multi-scale anisotropic definition) (use a
%function that returns vector based on position!)
%e_minor = [eigen_2-G;F];
%minor eigenvector (jkyprian image abstraction with coherence enhancing filtering definition)
%e_minor = [eigen_1-E;-F];

end