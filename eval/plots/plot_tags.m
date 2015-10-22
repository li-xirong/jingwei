function [ ] = plot_tags( tags_str, method_names, varargin )

    if nargin < 2
        error('At least one tagging output is needed.');
    end
    
    result_size = size(varargin{1});
    for i = 2:length(varargin)
        if result_size ~= size(varargin{i})
            error('All tagging output must have the same size.');
        end
        if size(varargin{i}) ~= length(tags_str)
            error('All tagging output must have the same number of tags as tags_str.');
        end
    end
    
    X = length(tags_str):-1:1;
    styles = {'*k','ok','>k','dk','^r','sk','.k','xk','vk','+k','<k','pk','hk'};
    
    %figure('units', 'normalized', 'position', [.1 .1 .7 .4]);
    %figure;
    set(gca, 'GridLineStyle', ':');
    set(gca, 'XColor', 'k');
    set(gca, 'YColor', 'k');
    set(gca, 'FontSize', 14);
    hold on;
    for i = 1:length(varargin)
        plot(varargin{i}, X, styles{i}, 'MarkerSize', 8);
    end

    grid on;
    box on;
    legend(method_names, 'Location', 'SouthEast');
    ylabel('Tags');
    xlabel('Average Precision');
    ylim([0,length(tags_str)+1]);

    y_ticks = fliplr(tags_str);
    y_ticks = {' ', y_ticks{:}, ' '};
    set(gca, 'YTickLabel', y_ticks);    
    set(gca, 'YTick', 0:length(tags_str)+1);

end

