%% crystal cropper crops out colored images of single crystals and superimposes them onto black backgrounds of 300 by 300 pixels
% Each single crystal image is tagged with the polarizer angle at which
% image acquisition was made and these images are saved in a folder.
% The name of the folder refers to the polymorphic form 

clear;clc;close all
picturenames=dir('*tif*');
[r,c]=size(picturenames);
size_croppedimage=300; % size of black background

crystalinformation_majaxis={};
crystalinformation_minaxis={};
crystalinformation_area={};
crystalinformation_ellipsearea={};
crystalinformation_ratio={};
 
% code counters
m=1; %m represents the m_th number of individual crystals detected
angle=0;%first angle of the polarizer

% user defined parameter
limit=0.8;% this limits sets the criteria to determine if the detected region consists of single crystal or overlapping aggregate

% the following section of the code extracts individual crystals from crystal ensembles and saves
% them into folders relating to the angle of the polarizer
directoryName='C:\Users\ariel\Desktop\glycinepolymorphs\alpha';
mkdir(directoryName);

angle_per_frame=input('Enter an angle (degree). How many degrees of rotation per frame? \n');
for l=1:r
tic
image=picturenames(l).name; 
I=rgb2gray(imread(image));%greyscale
C=imread(image);%colored rgb

%% thresholding using Otsu method to identify where crystals are
T=graythresh(I);
lambda=0.5;
BW = imbinarize(I,lambda*T);
%figure
%imshow(BW)

%% Identifies the region where crystals are detected
stats= regionprops('table',BW,'Area','BoundingBox','MajorAxisLength','MinorAxisLength','Orientation');
crystalinfo=table2array(stats);
crystalinfo_boundingbox=table2array(regionprops('table',BW,'BoundingBox'));
crystalinfo_area=table2array(regionprops('table',BW,'Area'));
crystalinfo_majoraxislength=table2array(regionprops('table',BW,'MajorAxisLength'));
crystalinfo_minoraxislength=table2array(regionprops('table',BW,'MinorAxisLength'));
crystalinfo_extent=table2array(regionprops('table',BW,'Extent'));
crystalinfo_orientation=table2array(regionprops('table',BW,'Orientation'));

%% bound all the crystals detected irregardless of whether they are aggregrates or single crystals, crop only single crystals out
[r1,c1]=size(crystalinfo);%r1 refers to number of regions detected where crystals are present
for j=1:r1
      %rectangle('position',crystalinfo_boundingbox(j,:),'LineWidth',1,'LineStyle','-','Edgecolor','r');     
      if crystalinfo_extent(j)*crystalinfo_boundingbox(j,3)*crystalinfo_boundingbox(j,4)/(3.1415/4*crystalinfo_majoraxislength(j)*crystalinfo_minoraxislength(j))>limit
          J=imcrop(C,crystalinfo_boundingbox(j,:));
          imagename1=sprintf('%07d.tif',m);
          fulldestination=fullfile(directoryName,imagename1);
          [r2,c2,d2]=size(J);%obtain size of cropped image
          blackbackground=zeros(size_croppedimage,size_croppedimage,3,'uint8');% size of black background image to which cropped single crystals will be superimposed upon
          for colorchannel=1:3
              blackbackground(1:r2,1:c2,colorchannel)=J(1:r2,1:c2,colorchannel);
              J1=blackbackground;
          end     
          imwrite(J1,fulldestination);
          singlecrystalsavedirectory(m,:)={fulldestination};
          singlecrystalboundingbox(m,[1:4])=stats.BoundingBox(j,:);
          singlecrystalminoraxis(m,:)=stats.MinorAxisLength(j,:);
          singlecrystalmajoraxis(m,:)=stats.MajorAxisLength(j,:);
          singlecrystalorientation(m,:)=stats.Orientation(j,:);
          singlecrystalarea(m,:)=stats.Area(j,:);
          singlecrystalpolarizerangle(m,:)=angle;
          m=m+1;
      end 
end
betaglycineDataset=table(singlecrystalsavedirectory, singlecrystalboundingbox,singlecrystalorientation,singlecrystalarea, singlecrystalminoraxis, singlecrystalmajoraxis, singlecrystalpolarizerangle);

%% Find out if area of ellipse matches the regionprops area, this will mean minor and major axis is a good approximation
ellipsearea=zeros(r1,1);
for k=1:r1
    ellipsearea(k,1)=3.14159*(crystalinfo_majoraxislength(k)/2)*(crystalinfo_minoraxislength(k)/2);
    ratio=ellipsearea./crystalinfo_area;
end
%% compiles the data of all crystal regions detected both overlapping and single crystals
crystalinformation_majaxis{l}=crystalinfo_majoraxislength;
crystalinformation_minaxis{l}=crystalinfo_minoraxislength;
crystalinformation_area{l}=crystalinfo_area;
crystalinformation_ellipsearea{l}=ellipsearea;
crystalinformation_ratio{l}=ratio;
crystalinformation=[{crystalinformation_majaxis},{crystalinformation_minaxis},{crystalinformation_area},{crystalinformation_ellipsearea},{crystalinformation_ratio}];
sprintf('image %d is completed!',l)
angle=angle+angle_per_frame;
if angle>360 %resets angle of polarizer back to 0 once revolution is complete
    angle=0;
end
toc
end
save('alpha.mat')
