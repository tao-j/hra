function [ndcg_array] = ndcg(gt, pred)
    [~, pred_argidx] = sort(pred, 'descend');
    [~, gt_argidx] = sort(gt, 'descend');
    assert(all(size(gt_argidx) == size(pred_argidx)));

    [len, len_2] = size(gt_argidx);
    assert(len > 1);
    assert(len_2 == 1);

    len_a = 1:len;
    len_a = len_a';

    IDCG = cumsum(gt(gt_argidx)./log2(len_a+1));
    DCG = cumsum(gt(pred_argidx)./log2(len_a+1));
    ndcg_array = DCG./IDCG;
    ndcg_array = ndcg_array(len);
end

