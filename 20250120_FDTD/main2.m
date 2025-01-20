% 2D FDTD Simulation for TMz Polarization

%%
% Clear workspace and close figures
clear; close all;

% Simulation parameters
Nx = 100;           % Number of cells in x-direction
Ny = 100;           % Number of cells in y-direction
dx = 1e-3;          % Spatial step in x (meters)
dy = 1e-3;          % Spatial step in y (meters)
dt = dx/(3e8*sqrt(2)); % Time step (Courant condition)
Nt = 1000;           % Number of time steps
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
f = 2e9;            % Source frequency (Hz)
omega = 2*pi*f;     % Angular frequency
sources = struct('i', {}, 'j', {}, 'A', {}, 'phi', {});
sources(1).i = 50;  % Source x-position
sources(1).j = 50;  % Source y-position
sources(1).A = 1;   % Amplitude
sources(1).phi = 0; % Phase (radians)

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

% Setup figure and GIF
fig = figure;
axis tight manual;
filename = 'fdtd_simulation.gif';

for n = 1:Nt
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
    
    % Plot and save frame
    imagesc(Hz');
    title(sprintf('Time Step: %d / %d', n, Nt));
    axis equal off;
    colormap('jet');
    colorbar;
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