% 2D FDTD Simulation for TMz Polarization with Final Field Plots and GIF Export

%%
% Clear workspace and close figures
clear; close all;

% Simulation parameters
Nx = 300;           % Number of cells in x-direction
Ny = 300;           % Number of cells in y-direction
dx = 1e-3;          % Spatial step in x (meters)
dy = 1e-3;          % Spatial step in y (meters)
% dt = dx/(3e8*sqrt(2)); % Time step (Courant condition)

Nt = 1000;           % Number of time steps
c0 = 3e8;           % Speed of light in vacuum
mu0 = pi*4e-7;      % Permeability of free space
eps0 = 8.85e-12;    % Permittivity of free space

dt = 1/(c0*sqrt(1/dx^2 + 1/dy^2)) * 0.5; % 99% of stability limit

% User-defined parameters
% 1. Refractive index distribution (Nx x Ny matrix)
n = ones(Nx, Ny);   % Initialize to vacuum
% Example: Add a circular obstacle
% [X, Y] = meshgrid(1:Nx, 1:Ny);
% n(sqrt((X-150).^2 + (Y-150).^2) < 50) = 2;

% 2. Source parameters
f = 10e9;           % Source frequency (Hz)
lambda = c0/f;      % Wavelength
omega = 2*pi*f;     % Angular frequency

% Create vertical line source (~3 wavelengths long)
source_length = round(0.3*lambda/dx); % Number of cells for 3 wavelengths
sources = struct('i', {}, 'j', {}, 'A', {}, 'phi', {});
center_x = round(Nx/2);
start_y = round(Ny/2 - source_length/2);
end_y = round(Ny/2 + source_length/2);

for j = start_y:end_y
    sources(end+1).i = center_x;
    sources(end).j = j;
    sources(end).A = 1;
    sources(end).phi = 0;
end

% Precompute material matrices
epsilon_r = n.^2;

% Create staggered grid permittivities
epsilon_Ex = zeros(Nx, Ny+1);
epsilon_Ey = zeros(Nx+1, Ny);

for i = 1:Nx
    for j = 1:Ny+1
        if j == 1
            epsilon_Ex(i,j) = epsilon_r(i,j);
        elseif j == Ny+1
            epsilon_Ex(i,j) = epsilon_r(i,Ny);
        else
            epsilon_Ex(i,j) = (epsilon_r(i,j) + epsilon_r(i,j-1))/2;
        end
    end
end

for i = 1:Nx+1
    for j = 1:Ny
        if i == 1
            epsilon_Ey(i,j) = epsilon_r(i,j);
        elseif i == Nx+1
            epsilon_Ey(i,j) = epsilon_r(Nx,j);
        else
            epsilon_Ey(i,j) = (epsilon_r(i,j) + epsilon_r(i-1,j))/2;
        end
    end
end

% Initialize field matrices
Ex = zeros(Nx, Ny+1);
Ey = zeros(Nx+1, Ny);
Hz = zeros(Nx, Ny);

% Precompute update coefficients
CEx = dt./(epsilon_Ex * dy);
CEy = -dt./(epsilon_Ey * dx);
CHz_x = dt/(mu0 * dx);
CHz_y = -dt/(mu0 * dy);

% First pass to determine maximum field values and final states
max_Ex = 1e-9;
max_Ey = 1e-9;
max_Hz = 1e-9;
final_fields = struct();

disp('Calculating field maxima and final states...');
tic;
for n = 1:Nt
    % Field updates

    % Vectorized Field Updates

    % Update Hz field (all rows/columns)
    % Calculate spatial derivatives using matrix operations
    dEy_dx = Ey(2:Nx+1, :) - Ey(1:Nx, :);    % Forward difference in x
    dEx_dy = Ex(:, 2:Ny+1) - Ex(:, 1:Ny);    % Forward difference in y
    Hz = Hz + CHz_x * dEy_dx + CHz_y * dEx_dy;
    
    % Update Ex field (all rows, columns 2:end)
    Hz_diff_x = Hz(:, 2:Ny) - Hz(:, 1:Ny-1);
    Ex(:, 1:Ny-1) = Ex(:, 1:Ny-1) + CEx(:, 2:Ny) .* Hz_diff_x;
    
    % Update Ey field (rows 2:end, all columns)
    Hz_diff_y = Hz(2:Nx, :) - Hz(1:Nx-1, :);
    Ey(1:Nx-1, :) = Ey(1:Nx-1, :) + CEy(2:Nx, :) .* Hz_diff_y;
    
    
    % Apply sources
    for s = 1:length(sources)
        i = sources(s).i;
        j = sources(s).j;
        A = sources(s).A;
        phi = sources(s).phi;
        Hz(i,j) = Hz(i,j) + A * sin(omega*(n + 0.5)*dt + phi);
    end
    
    % Calculate field magnitudes for visualization
    Ex_plot = (Ex(:,1:end-1) + Ex(:,2:end))/2;
    Ey_plot = (Ey(1:end-1,:) + Ey(2:end,:))/2;
    Hz_plot = Hz;
    
    % Update maxima
    max_Ex = max(max_Ex, max(abs(Ex_plot(:))));
    max_Ey = max(max_Ey, max(abs(Ey_plot(:))));
    max_Hz = max(max_Hz, max(abs(Hz_plot(:))));
    
    % Store final fields
    if n == Nt
        final_fields.Ex = Ex_plot;
        final_fields.Ey = Ey_plot;
        final_fields.Hz = Hz_plot;
    end
end
toc;

% Plot final field distributions
fig_final = figure('Position', [100 100 1200 400]);
subplot(1,3,1);
imagesc(final_fields.Ex');
clim([-max_Ex max_Ex]);
title('Final E_x Field');
axis equal tight off;
colorbar;

subplot(1,3,2);
imagesc(final_fields.Ey');
clim([-max_Ey max_Ey]);
title('Final E_y Field');
axis equal tight off;
colorbar;

subplot(1,3,3);
imagesc(final_fields.Hz');
clim([-max_Hz max_Hz]);
title('Final H_z Field');
axis equal tight off;
colorbar;

colormap(jet);
sgtitle('Final Field Distributions');
drawnow;

%%
% Second pass for GIF generation
Ex = zeros(Nx, Ny+1);
Ey = zeros(Nx+1, Ny);
Hz = zeros(Nx, Ny);

fig_gif = figure('Position', [100 100 2000 600]);
filename = 'fdtd_3components.gif';

disp('Generating animation...');
tic;
for n = 1:Nt
    % Field updates

    % Vectorized Field Updates

    % Update Hz field (all rows/columns)
    % Calculate spatial derivatives using matrix operations
    dEy_dx = Ey(2:Nx+1, :) - Ey(1:Nx, :);    % Forward difference in x
    dEx_dy = Ex(:, 2:Ny+1) - Ex(:, 1:Ny);    % Forward difference in y
    Hz = Hz + CHz_x * dEy_dx + CHz_y * dEx_dy;

    % Update Ex field (all rows, columns 2:end)
    Hz_diff_x = Hz(:, 2:Ny) - Hz(:, 1:Ny-1);
    Ex(:, 2:Ny) = Ex(:, 2:Ny) + CEx(:, 2:Ny) .* Hz_diff_x;
    
    % Update Ey field (rows 2:end, all columns)
    Hz_diff_y = Hz(2:Nx, :) - Hz(1:Nx-1, :);
    Ey(2:Nx, :) = Ey(2:Nx, :) + CEy(2:Nx, :) .* Hz_diff_y;
    
    
    % Apply sources
    for s = 1:length(sources)
        i = sources(s).i;
        j = sources(s).j;
        A = sources(s).A;
        phi = sources(s).phi;
        Hz(i,j) = Hz(i,j) + A * sin(omega*(n + 0.5)*dt + phi);
    end
    
    % Calculate field magnitudes for visualization
    Ex_plot = (Ex(:,1:end-1) + Ex(:,2:end))/2;
    Ey_plot = (Ey(1:end-1,:) + Ey(2:end,:))/2;
    Hz_plot = Hz;
    
    % Plot all components
    figure(fig_gif);
    subplot(1,3,1);
    imagesc(Ex_plot');
    clim([-max_Ex max_Ex]);
    title(sprintf('E_x @ step %d/%d', n, Nt));
    axis equal tight off;
    
    subplot(1,3,2);
    imagesc(Ey_plot');
    clim([-max_Ey max_Ey]);
    title(sprintf('E_y @ step %d/%d', n, Nt));
    axis equal tight off;
    
    subplot(1,3,3);
    imagesc(Hz_plot');
    clim([-max_Hz max_Hz]);
    title(sprintf('H_z @ step %d/%d', n, Nt));
    axis equal tight off;
    
    colormap(jet);
    drawnow;
    
    % Export to GIF
    frame = getframe(fig_gif);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);
    if n == 1
        imwrite(imind, cm, filename, 'gif', 'Loopcount', inf, 'DelayTime', 0.1);
    else
        imwrite(imind, cm, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
    end
end
toc;
disp('Simulation complete!');