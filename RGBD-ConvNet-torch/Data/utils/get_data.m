function get_data()

filename = 'rgbd-dataset';
fileFolder=fullfile('..', filename);
dirOutput=dir(fullfile(fileFolder,'*'));
classes={dirOutput.name}';
classes=classes(3:end,:);
lenClass = length(classes);

Size = 256;
mkdir(fullfile('..', 'data')) ;


for class = 1:lenClass
	fprintf('class: %d\n', class);
	for object = 1:1:15
		objectName = strcat(classes{class},'_', num2str(object));
		PathObject = fullfile('..', 'rgbd-dataset',classes{class},objectName);
		if  exist(PathObject)
			cnt = 1;
			for k = [1 2 4]
				for num = 1:5:2001
										
			             	RGBimageName = strcat(objectName,'_',num2str(k),'_',num2str(num),'_','crop.png');
					PathRGB = fullfile(PathObject,RGBimageName);
					DepthimageName = strcat(objectName,'_',num2str(k),'_',num2str(num),'_','depthcrop.png');
					PathDepth = fullfile(PathObject,DepthimageName);
								
					if ~exist(PathDepth)
						break;
					else
						%RGB
						im = im2single(process(imread(PathRGB), Size));
						RGB(1,:,:) = im(:,:,1);
						RGB(2,:,:) = im(:,:,2);
						RGB(3,:,:) = im(:,:,3);

						%SN
						Depth = process(single(imread(PathDepth)), Size);
						[SN_1 SN_2 SN_3] = surfnorm(Depth);
						SN(1,:,:)=SN_1;
						SN(2,:,:)=SN_2;
						SN(3,:,:)=SN_3;

						imdb.RGB = RGB;
						imdb.SN = SN;
						imdb.target = class;
						DataDir = fullfile('..', 'data', strcat(objectName,'_',num2str(cnt),'.mat'));						
						save(DataDir, '-struct', 'imdb', '-v7.3');																 
											 					 
						cnt = cnt+1;
					end
				end
			end
			fprintf('  %s\n', objectName);    
		end   
	end 
end
return




function image = process(im,Size)

	[h w d] = size(im);

	if h>w
		tmp = floor(w*Size/h);
		im = imresize(im,[Size tmp],'bicubic');
		a = floor((Size-tmp)/2);
		b = ceil((Size-tmp)/2); 
		
		for i = 1:1:d
			left = im(:,1,i);
			right = im(:,end,i);
			image(:,:,i) = [repmat(left,1,a), im(:,:,i), repmat(right,1,b)];
		end
	else
		tmp = floor(h*Size/w);
		im = imresize(im,[tmp Size],'bicubic');
		a = floor((Size-tmp)/2);
		b = ceil((Size-tmp)/2); 
		
		for i = 1:1:d
			top = repmat(im(1,:,i),a,1);
			middle = im(:,:,i);
			bottom = repmat(im(end,:,i),b,1);
			image(:,:,i) = [top; middle; bottom];
		end    
	end
return