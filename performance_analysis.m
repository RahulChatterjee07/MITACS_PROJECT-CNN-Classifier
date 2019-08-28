filename = 'F:\final_motortask.txt'; % experiment design
delimiter = '\t';

%% Format string for each line of text:
%   column1: double (%f)
%	column2: double (%f)
%   column3: double (%f)
%	column4: double (%f)
%   column5: double (%f)
%	column6: double (%f)
%   column7: double (%f)
%	column8: double (%f)
%   column9: double (%f)
% For more information, see the TEXTSCAN documentation.
formatSpec = '%f%f%f%f%f%f%f%f%f%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to format string.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter,  'ReturnOnError', false);

%% Close the text file.
fclose(fileID);

%% Post processing for unimportable data.
% No unimportable data rules were applied during the import, so no post
% processing code is included. To generate code which works for
% unimportable data, select unimportable cells in a file and regenerate the
% script.

%% Allocate imported array to column variable names
VarName1 = dataArray{:, 1};
VarName2 = dataArray{:, 2};
VarName3 = dataArray{:, 3};
VarName4 = dataArray{:, 4};
VarName5 = dataArray{:, 5};
VarName6 = dataArray{:, 6};
VarName7 = dataArray{:, 7};
VarName8 = dataArray{:, 8};
VarName9 = dataArray{:, 9};


%% Initialize variables.
filename = 'F:\subject4_motor1.txt'; %force output file
delimiter = '\t';

%% Read columns of data as strings:
% For more information, see the TEXTSCAN documentation.
formatSpec = '%s%s%s%s%s%s%s%s%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to format string.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter,  'ReturnOnError', false);

%% Close the text file.
fclose(fileID);

%% Convert the contents of columns containing numeric strings to numbers.
% Replace non-numeric strings with NaN.
raw = repmat({''},length(dataArray{1}),length(dataArray)-1);
for col=1:length(dataArray)-1
    raw(1:length(dataArray{col}),col) = dataArray{col};
end
numericData = NaN(size(dataArray{1},1),size(dataArray,2));

for col=[1,2,3,4,5,6,7,8]
    % Converts strings in the input cell array to numbers. Replaced non-numeric
    % strings with NaN.
    rawData = dataArray{col};
    for row=1:size(rawData, 1);
        % Create a regular expression to detect and remove non-numeric prefixes and
        % suffixes.
        regexstr = '(?<prefix>.*?)(?<numbers>([-]*(\d+[\,]*)+[\.]{0,1}\d*[eEdD]{0,1}[-+]*\d*[i]{0,1})|([-]*(\d+[\,]*)*[\.]{1,1}\d+[eEdD]{0,1}[-+]*\d*[i]{0,1}))(?<suffix>.*)';
        try
            result = regexp(rawData{row}, regexstr, 'names');
            numbers = result.numbers;
            
            % Detected commas in non-thousand locations.
            invalidThousandsSeparator = false;
            if any(numbers==',');
                thousandsRegExp = '^\d+?(\,\d{3})*\.{0,1}\d*$';
                if isempty(regexp(numbers, thousandsRegExp, 'once'));
                    numbers = NaN;
                    invalidThousandsSeparator = true;
                end
            end
            % Convert numeric strings to numbers.
            if ~invalidThousandsSeparator;
                numbers = textscan(strrep(numbers, ',', ''), '%f');
                numericData(row, col) = numbers{1};
                raw{row, col} = numbers{1};
            end
        catch me
        end
    end
end


%% Replace non-numeric cells with NaN
R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),raw); % Find non-numeric cells
raw(R) = {NaN}; % Replace non-numeric cells

%% Allocate imported array to column variable names
Time = cell2mat(raw(:, 1));
Task = cell2mat(raw(:, 2));
AnalogInput = cell2mat(raw(:, 3));
SummedSensors = cell2mat(raw(:, 4));
VarName5 = cell2mat(raw(:, 5));
Empty = cell2mat(raw(:, 6));
Empty1 = cell2mat(raw(:, 7));
Empty2 = cell2mat(raw(:, 8));

mvc=22.5; %set mvc value here 
E=[];

%finding the index for the task events
for i=1:length(SummedSensors)-2
   
    if Task(i+1)==0 &&Task(i+2)==mvc
        
        
        E(i)=(i+2);
        
    end
end
E(E==0)=[];
F =E+201;
% error measurement

r=[];
for i=1:75
    t=SummedSensors(E(i):F(i));
    u=find(t>=mvc);
    r(i)=min(u);
end
r=r+E;
r=r-1;
r(r==0);
o=[];


for i=1:75
    p=SummedSensors(r(i):F(i)-2);
    p=abs(p-mvc);
    o(i)=sum(p)/length(p);
end

R=[];
L=[];
P=[];
for i = 1:75

if (VarName3(2*i)==0.75)
    L(i)=i;

elseif (VarName3(2*i)==0.5) 
    P(i)=i;
end
end
L(L==0) = [];
P(P==0) = [];
for j = 1:25
    L(j)=o(L(j));
end
for j = 1:25
    P(j)=o(P(j));
end
T=sum(o)-sum(L)-sum(P);
err_1=(T/25);
err_2=(sum(L)/25);
err_3=(sum(P)/25);

%reaction time measurement
response_time=20*(r-E);
R1=[];
L1=[];
P1=[];
for i = 1:75

if (VarName3(2*i)==0.75)
    L1(i)=i;

elseif (VarName3(2*i)==0.5) 
    P1(i)=i;
end
end
L1(L1==0) = [];
P1(P1==0) = [];

for j = 1:25
    L1(j)=response_time(L1(j));
end
for j = 1:25
    P1(j)=response_time(P1(j));
end
T1=sum(response_time)-sum(L1)-sum(P1);
res_1=(T1/25);
res_2=(sum(L1)/25);
res_3=(sum(P1)/25);

fprintf('error for gain 1 is %.2f\n',err_1);
fprintf('error for gain 0.75 is %.2f\n',err_2);
fprintf('error for gain 0.5 is %.2f\n',err_3);

fprintf('response time for gain 1 is %.f\n',res_1);
fprintf('response time for gain 0.75 is %.f\n',res_2);
fprintf('response time for gain 0.5 is %.f\n',res_3);
    
    
    
    