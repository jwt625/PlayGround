%%
clc;
% Initialize parameters
 clear all;
 close all;

% Grid size (assuming Nx = Ny = 100 for simplicity)
Nx = 100; % Number of grid points in x-direction
Ny = 100; % Number of grid points in y-direction

% Mesh spacing (uniform grid)
dx = 2*pi / (Nx-1);
dy = dx; 

% Time steps
Nt = 500;

% Speed of light
c = 1;
omega = 2*c*Nt/(Nx*dx); % Angular frequency

% Source parameters
src_pos = [Nx/2, Ny/2]; % Central point source position in grid indices (assuming Nx and Ny are even)
amp = 1; % Amplitude at the source
phase = 0; % Phase of the source
time_shift = 0:1:Nt-1;

% Initialize fields
E_x = zeros(Ny, Nx); % Ex field component
E_y = zeros(Ny, Nx); % Ey field component
H_z = zeros(Ny, Nx); % Hz field component

% Constitutive parameter (refractive index squared)
eps_r = ones(Ny, Nx);

% Perfectly Matched Layer (PML) parameters - Simplified implementation for demonstration purposes
pml_width = 10; % Number of grid points in PML

% Initialize FDTD grid with PML boundaries
E_x = padarray(E_x, [pml_width, pml_width], 'both');
E_y = padarray(E_y, [pml_width, pml_width], 'both');
H_z = padarray(H_z, [pml_width, pml_width], 'constant', 'both');

% Time stepping variables
omega = 2*pi*c*Nt/(Nx*dx); % Angular frequency

% FDTD main loop
for t = 1:Nt
    % Update E fields (Yee's algorithm for TE polarization)
    E_y(:, :, 2) = E_y(:, :, 2) .* circshift(H_z, [0, -1]);
    
    H_z = (omega/eps_r) .* circshift(E_y, [1, 0]) - ...
         ((omega*dy/2).^-1) * (E_y - circshift(E_y, [-1, 0]));
    
    E_x(:, :, 2) = E_x(:, :, 2) .* circshift(H_z, [0, -1]);
    
    H_z = (omega/eps_r) .* circshift(E_x, [1, 0]) - ...
         ((omega*dx/2).^-1) * (E_x - circshift(E_x, [-1, 0]));
    
    % Update E fields
    E_y(:, :, 1) = E_y(:, :, 1) .* circshift(H_z, [0, -1]);
    
    H_z = (omega/eps_r) .* circshift(E_y, [1, 0]) - ...
         ((omega*dy/2).^-1) * (E_y - circshift(E_y, [-1, 0]));
    
    E_x(:, :, 1) = E_x(:, :, 1) .* circshift(H_z, [0, -1]);
    
    % Apply boundary conditions with PML
    H_z(1+pml_width, :) = exp(-1i*omega*t); % Simplified PML implementation
    
    % Inject source excitation at the center position
    if src_pos(1) == current_x && src_pos(2) == current_y
        E_y(:, :, : ) = amp * exp(1i*(omega*t + phase));
    end

    % Store field data for visualization
    store_E_y = E_y;
    
    % Update time step variables
    E_y = E_y';
    H_z = H_z';
end

% Visualize the simulation results
figure;
for t = 1:Nt
    imagesc(abs(store_E_y(:,:,t)));
    title(sprintf('Time Step: %d', t));
    drawnow;
end

% Export animation as GIF (not recommended due to memory constraints, but for demonstration purposes)
try
    videoWriter = VideoWriter('FDTD_WavePropagation.mp4', 'avi');
    open(videoWriter);
    
    for t = 1:Nt
        imagesc(abs(store_E_y(:,:,t)));
        title(sprintf('Time Step: %d', t));
        drawnow;
        
        frame = getframe(gcf, [0.5 0.5 1 1]);
        videoWriter.writeVideo(frame);
    end
    
    close(videoWriter);
    close(gcf);
catch
    disp('Error exporting animation as GIF.');
end
