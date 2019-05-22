GUMBEL = 'g';
NORMAL = 'n';

dist_flags = [GUMBEL, NORMAL];
adv_flags = [true, false];

% parpool(4);
parfor i = 1:4
    if i == 1
        main(GUMBEL, true);
    elseif i == 2
        main(NORMAL, true);
    elseif i == 3
        main(GUMBEL, false);
    elseif i ==4
        main(NORMAL, false);
    end
end
