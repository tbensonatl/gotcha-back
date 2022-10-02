function [img] = show_image(image_filename)

    fp = fopen(image_filename, 'rb');
    img = fread(fp, inf, 'single');
    fclose(fp);

    img = complex(img(1:2:end),img(2:2:end));
    Nx = sqrt(numel(img));

    if Nx*Nx ~= numel(img), error('This script assumes square images.'); end

    Ny = Nx;
    img = reshape(img, [Nx Ny]);
    img = img.';

    imdim = 150;
    ix = linspace(-100,100-imdim/1024,1024);
    iy = linspace(-100,100-imdim/1024,1024);

    figure; imagesc(ix,iy,20*log10(abs(img./max(abs(img(:))))));

    axis xy;
    colorbar;
    colormap gray;
    caxis([-80 0]);

    title('Pixel magnitude in normalized decibel units')
end
