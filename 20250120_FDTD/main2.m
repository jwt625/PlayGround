% 2D FDTD Simulation for TMz Polarization with Improved Visualization

%%
% Clear workspace and close figures
clear; close all;

% Simulation parameters
Nx = 300;           % Number of cells in x-direction
Ny = 300;           % Number of cells in y-direction
dx = 3e-3;          % Spatial step in x (meters)
dy = 3e-3;          % Spatial step in y (meters)
dt = dx/(3e8*sqrt(2)); % Time step (Courant condition)
Nt = 300;           % Number of time steps
c0 = 3e8;           % Speed of light in vacuum
mu0 = pi*4e-7;      % Permeability of free space
eps0 = 8.85e-12;    % Permittivity of free space

% User-defined parameters
% 1. Refractive index distribution (Nx x Ny matrix)
n = ones(Nx, Ny);   % Initialize to vacuum
% Example: Add a circular obstacle
% [X, Y] = meshgrid(1:Nx, 1:Ny);
% n(sqrt((X-50).^2 + (Y-50).^2) < 20) = 2;

% 2. Source parameters
f = 10e9;           % Source frequency (Hz)
lambda = c0/f;      % Wavelength
omega = 2*pi*f;     % Angular frequency

% Create vertical line source (~10 wavelengths long)
source_length = round(3*lambda/dx); % Number of cells for 10 wavelengths
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

% First pass to determine maximum field values
max_Ex = 1e-9;
max_Ey = 1e-9;
max_Hz = 1e-9;

disp('Calculating field maxima...');
tic;
for n = 1:Nt
    % Field updates
    % Update Ex field
    for i = 1:Nx
        for j = 2:Ny
            Ex(i,j) = Ex(i,j) + CEx(i,j) * (Hz(i,j) - Hz(i,j-1));
        end
    end
    % Update Ey field
    for i = 2:Nx
        for j = 1:Ny
            Ey(i,j) = Ey(i,j) + CEy(i,j) * (Hz(i,j) - Hz(i-1,j));
        end
    end
    % Update Hz field
    for i = 1:Nx
        for j = 1:Ny
            dEy_dx = Ey(i+1,j) - Ey(i,j);
            dEx_dy = Ex(i,j+1) - Ex(i,j);
            Hz(i,j) = Hz(i,j) + CHz_x * dEy_dx + CHz_y * dEx_dy;
        end
    end
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
end
toc;
% Second pass for visualization
Ex = zeros(Nx, Ny+1);
Ey = zeros(Nx+1, Ny);
Hz = zeros(Nx, Ny);

fig = figure('Position', [100 100 2000 600]);
filename = 'fdtd_3components.gif';

disp('Generating visualization...');
tic;
for n = 1:Nt
    % Field updates
    
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
    
    % Plot all three components
    subplot(1,3,1);
    imagesc(Ex_plot');
    clim([-max_Ex max_Ex]);
    title('E_x Component');
    axis equal tight;
    colorbar
    
    subplot(1,3,2);
    imagesc(Ey_plot');
    clim([-max_Ey max_Ey]);
    title('E_y Component');
    axis equal tight;
    colorbar
    
    subplot(1,3,3);
    imagesc(Hz_plot');
    clim([-max_Hz max_Hz]);
    title('H_z Component');
    axis equal tight;
    colorbar
    
    colormap(jet);
    sgtitle(sprintf('Time Step: %d / %d', n, Nt));
    drawnow;
    
    % Capture and save frame to GIF
    frame = getframe(fig);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);
    if n == 1
        imwrite(imind, cm, filename, 'gif', 'Loopcount', inf, 'DelayTime', 0.1);
    else
        imwrite(imind, cm, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
    end
end
toc;