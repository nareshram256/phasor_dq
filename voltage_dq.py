import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
frequency = 0.05  # Frequency in Hz
omega = 2 * np.pi * frequency  # Angular frequency in rad/s

# Clarke and Park Transform Functions
def clarke_transform(a, b, c):
    ialpha = (2 * a - b - c) / 3
    ibeta = (np.sqrt(3) * (b - c)) / 3
    return ialpha, ibeta

def park_transform(ialpha, ibeta, theta2):
    id = ialpha * np.exp(1j*theta2) - ibeta * np.exp(1j*theta2)
    iq = -ialpha * np.exp(1j*theta2)*1j + ibeta * np.exp(1j*theta2)
    return id, iq

# Create a figure with two subplots
fig, (ax_polar, ax_cartesian,ax_cartesian3) = plt.subplots(1, 3, figsize=(12, 6))

# Set up the polar subplot
ax_polar = plt.subplot(1, 3, 1, projection='polar')
ax_polar.set_ylim(0, 1)
ax_polar.set_yticklabels([])
ax_polar.grid(True)
ax_polar.set_title('three phasors and res')
# Set up the polar subplot
#ax_polar2 = plt.subplot(1, 3, 2, projection='polar')
#ax_polar2.set_ylim(0, 1)
#ax_polar2.set_yticklabels([])
#ax_polar2.grid(True)
#ax_polar2.set_title(r'$\alpha$-$\beta$ axis')

# Set up the polar subplot
#ax_polar3 = plt.subplot(1, 3, 3, projection='polar')
#ax_polar3.set_ylim(0, 1)
#ax_polar3.set_yticklabels([])
#ax_polar3.grid(True)
#ax_polar3.set_title('D-Q axis')

# Set up the Cartesian subplot
ax_cartesian = plt.subplot(1, 3, 2)
ax_cartesian.set_xlim(-1, 1)
ax_cartesian.set_ylim(-1, 1)
ax_cartesian.set_xlabel('Real')
ax_cartesian.set_ylabel('Imaginary')
ax_cartesian.grid(True)
ax_cartesian.set_title(r'$\alpha$-$\beta$ axis')


# Set up the Cartesian subplot
ax_cartesian3 = plt.subplot(1, 3, 3)
ax_cartesian3.set_xlim(-1, 1)
ax_cartesian3.set_ylim(-1, 1)
ax_cartesian3.set_xlabel('Real')
ax_cartesian3.set_ylabel('Imaginary')
ax_cartesian3.grid(True)
ax_cartesian3.set_title('D-Q axis')

# Initialize the phasors
phasors_polar = [ax_polar.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color=c) for c in ['r', 'g', 'b']]
phasor_clarke = [ax_cartesian.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color=c, label='Clarke Transform') for c in ['r','g','black']]
#phasor_park = [ax_polar3.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color=c, label='Park Transform') for c in ['r','g','black']]
phasor_res = ax_polar.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color='black', label='Res Transform')
phasor_park = [ax_cartesian3.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color=c, label='Park Transform') for c in ['r','g','black']]


def update(frame):
    t = frame  # Convert frame number to time in seconds
    # Original Phasors
    phasors_real = []
    phasors_imag = []
    artists = []
    V=[]
    theta=np.pi/6
    mag=[1,1,1]
    for k, phasor in enumerate(phasors_polar):
        phase_angle = omega * t+theta- k * 2 * np.pi / 3  # 120 degrees phase shift for each phasor
        volt=mag[k]*np.exp(1j*phase_angle)
        V.append(volt)
        #phasor_new = 1 * np.exp(1j * phase_angle)
        phasor.set_UVC(k * 2 * np.pi / 3, np.real(volt))
        artists.append(phasor)
    Vdc=V[0]+V[1]+V[2]
    phasor_res.set_UVC(omega * t+theta*t, np.real(Vdc))
    artists.append(phasor_res)
    # Clarke and Park Transforms
    for k, phasor2 in enumerate(phasor_clarke ):
        if(k!=2):
            phasor2.set_UVC(k, 1-k)
        else:
            ialpha, ibeta = clarke_transform(V[0],V[1], V[2])
            ang=np.arctan2(np.imag(ialpha), np.real(ialpha))
            phasor2.set_UVC(np.real(ialpha),np.imag(ialpha))
        artists.append(phasor2)
        
    #rotating angle required for DQ   
    
    for k, phasor3 in enumerate(phasor_park):
        id, iq = park_transform(ialpha, ibeta, -ang) # ang is time varying quantity
        #print(angle_degrees = np.degrees(np.arctan2(iq, id)))
        ang2=np.arctan2(np.imag(id), np.real(id))
        if(k==0):
            phasor3.set_UVC(np.cos(ang2),np.sin(ang2))
        if(k==1):
            phasor3.set_UVC(-np.sin(ang2),np.cos(ang2))
        else:
            phasor3.set_UVC(np.real(id),np.imag(id)) # id rotating
        artists.append(phasor3)
    return artists
# Creating the animation
ani = FuncAnimation(fig, update, frames=np.linspace(1, 60, 300), blit=True)

# Uncomment the following line to save the animation as an mp4 file
#ani.save('rotating_phasors_with_transforms.mp4', fps=60, extra_args=['-vcodec', 'libx264'])

# Save the animation as a GIF file using ImageMagick
ani.save('rotating_phasors_with_transforms.gif', writer='imagemagick', fps=60)

plt.show()