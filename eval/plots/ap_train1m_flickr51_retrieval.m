function per_concept

datafile = 'flickr51.ap.txt';
[concepts, tagpos, sf, tagvote, tagprop, relexample] = textread(datafile, '%s %f %f %f %f %f');

data = [tagpos, sf, tagvote, tagprop, relexample];
[row,column] = size(data);

baseline = relexample;
[temp, ix] = sort(baseline, 'ascend');
data = data(ix, :);
concepts = concepts(ix);

markers = {'ko', 'ks', 'k^', 'kx', 'r>', 'd', 'rs', 'rx', 'r>', 'ro'};
%%
% Plotting
markersize = 10;
fontsize = 12;
%fontsize_legend = 14;

g = [];

i=1;
g(i) = plot(data(:,i), 1:row, markers{i}, 'MarkerSize', markersize, 'LineWidth', 1.5);
hold on;



for i=2:column,    
    g(i) = plot(data(:,i), 1:row, markers{i}, 'MarkerSize', markersize, 'LineWidth', 1.5);
end,


%for i=1:row,
%    x = 0:0.01:baseline(i);
%    plot(x, i*ones(1, length(x)), 'g-');
%end

%%
set(gca,'YTick', 1:length(concepts), 'YTickLabel', concepts);
xlabel('Average precision', 'FontName', 'arial', 'FontSize', fontsize, 'FontWeight', 'bold');
ylabel('Test tags', 'FontName', 'arial', 'FontSize', fontsize, 'FontWeight', 'bold')
set(gca, 'FontSize', fontsize);
set(gcf,'color','white')

%set(gcf,'Position',[50 0 700 1150]);
set(gcf,'Position',[50 0 800 800]);
%set(gcf,'PaperPositionMode','manual');
set(gca,'XGrid','on');
set(gca,'YGrid','on');

set(gca, 'YLim', [0 length(concepts)+1]);
set(gca, 'XLim', [0, 1]);
%%
%set(legend, 'interpreter', 'latex')
legend({'TagPosition', 'SemanticField', 'TagVote', 'TagProp', 'RelExample'});
set(legend, 'FontSize', fontsize+2, 'Location', 'NorthWest');

pos = get(gca, 'Position');
%pos(2) = 0.05; % bottom margin
pos(2) = 0.06;
pos(3) = 0.85; % right margin
%pos(4) = 0.85; 
pos(4) = 0.93; % top margin
set(gca, 'Position', pos);
saveTightFigure('flickr51-ap.pdf');


